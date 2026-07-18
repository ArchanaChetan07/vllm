# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVCacheSpec + SingleTypeKVCacheManager for MiniCPM-SALA's InfLLM-V2
sparse attention layers.

STAGE 3 SCOPE (see docs/minicpm_sala_known_limitations.md):

Stage 3a (`HierarchicalCompressedAttentionSpec`): the declarative "what
does this layer's cache look like" description, registered via
`vllm/v1/kv_cache_spec_registry.py`'s `@register_kv_cache_spec` -- a
first-class, documented extension point ("Out-of-tree platforms can
define custom specs and managers..."), not a hardcoded dict edit as
originally assumed in the Phase 2/3 design doc.

Stage 3b (`HierarchicalCompressedAttentionManager`): the cache-hit /
block-allocation logic. Written directly against `MambaManager`'s real
implementation as precedent (`vllm/v1/core/single_type_kv_cache_manager.py`)
rather than invented. The key design decision -- **prefix-cache-hit
reuse is disabled for this cache type** (`get_num_common_prefix_blocks`
returns 0, `find_longest_cache_hit` returns empty) -- mirrors
`MambaManager.get_num_common_prefix_blocks`'s own real docstring
("cascade attention is not supported by mamba") exactly, and for the
same underlying reason: InfLLM-V2's incremental, per-token compression
state (Phase 1 report §2b/§4 -- the `no_compress_k`/`no_compress_k2`
ring-buffer staging that only emits a new compressed row every
`kernel_stride` tokens) is sequentially stateful per-request, not a
block-hash-addressable structure that can be safely shared or resumed
across different requests the way a plain full-attention KV block can.
Treating it as prefix-cacheable without proving that safe would risk
silent cross-request cache corruption -- exactly class of bug flagged
as landmine #38643 in the Phase 2/3 design doc. This is the same
conservative choice Mamba's own authors made for an analogous reason,
not a corner cut for expedience.

NOT written: the more elaborate `mamba_cache_mode="align"` /
segment-boundary sparse-retention machinery `MambaManager` optionally
supports. That is a genuine performance optimization (allowing SOME
prefix-cache reuse at aligned boundaries) layered on top of the
conservative baseline this file implements; worth revisiting once the
baseline is confirmed correct against a real scheduler, not before.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec


def build_hierarchical_compressed_attention_spec(
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    compress_kernel_size: int,
    compress_kernel_stride: int,
    dense_len: int,
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
) -> "HierarchicalCompressedAttentionSpec":
    """Build the scheduler KV spec for InfLLM-V2 sparse attention layers."""
    return HierarchicalCompressedAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        compress_kernel_size=compress_kernel_size,
        compress_kernel_stride=compress_kernel_stride,
        dense_len=dense_len,
        kv_quant_mode=kv_quant_mode,
    )


if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock


@dataclass(frozen=True, kw_only=True)
class HierarchicalCompressedAttentionSpec(AttentionSpec):
    """KV cache spec for InfLLM-V2 sparse attention layers (the
    "minicpm4" mixer type). Inherits from `AttentionSpec` (not bare
    `KVCacheSpec`) because the dominant memory term IS a full K/V
    cache, identical in kind to `FullAttentionSpec` -- see the Phase 1
    report's §2c correction: the compression tiers are additive
    overhead for cheap top-k block selection, not a replacement for
    full-resolution storage. `AttentionSpec` already provides
    `num_kv_heads`, `head_size`, `dtype` -- reused here rather than
    redeclared, since duplicating them would violate the "no duplicated
    logic" engineering rule for zero actual benefit.

    Extra fields describe the two additional compression-tier terms
    (tier-1 and tier-2; no separate staging-buffer term -- see
    `page_size_bytes`'s real design note for why),
    all derived from `MiniCPMInfLLMv2Attention`'s real constructor
    arguments (`sparse_config` in the HF config) -- see Phase 1 report
    §2b for the exact reference-code provenance of each.
    """

    # From config.sparse_config -- tier-1 compression (kernel_size=32,
    # kernel_stride=16 in the released checkpoint).
    compress_kernel_size: int
    compress_kernel_stride: int
    # Tier-2 compression is always 4x tier-1 in the reference
    # implementation (`CompressK(..., kernel_size=self.kernel_size * 4,
    # kernel_stride=self.kernel_stride * 4)` in
    # `MiniCPMInfLLMv2Attention.__init__`) -- not an independent
    # parameter, so not redeclared as one here; derived via properties
    # below instead, to make that 4x relationship structurally
    # impossible to drift out of sync with the reference.
    # Below this context length, the reference model (and this port's
    # Stage-1 dense fallback) doesn't use the sparse path at all.
    dense_len: int

    @property
    def compress_k2_kernel_size(self) -> int:
        return self.compress_kernel_size * 4

    @property
    def compress_k2_kernel_stride(self) -> int:
        return self.compress_kernel_stride * 4

    @property
    def page_size_bytes(self) -> int:
        """Bytes for one `block_size`-token page.

        REVISED (final design, made while wiring up the actual sparse
        forward path in `vllm/v1/attention/backends/minicpm_sala_sparse.py`):
        this now returns ONLY the full-K/V term, identical to what a
        plain `FullAttentionSpec` would report. Earlier passes of this
        file included additional tier-1/tier-2 compressed-K terms here,
        under the assumption that compressed tiers would be persisted
        in vLLM's managed cache across decode steps. The final
        implementation instead recomputes compressed tiers FRESH on
        every call, directly from the full K cache (see
        `_forward_sparse`'s module-level design note for the full
        reasoning -- this is the "measure before optimizing" tradeoff
        flagged since this file's first version). Since nothing related
        to compression is actually stored in vLLM's page-allocated
        memory, `page_size_bytes` MUST match that reality exactly --
        `AttentionBackend.get_kv_cache_shape` (the actual physical
        allocation vLLM performs) and this property must always agree,
        or vLLM's memory planner will account for GPU memory wrongly. Kept
        as an explicit property (rather than just inheriting
        `AttentionSpec`'s default) so that IF persistent tier caching is
        added later as a real optimization, this is the one place that
        needs to grow again, symmetric with `get_kv_cache_shape`.
        """
        return self._full_kv_page_size_bytes

    @property
    def _full_kv_page_size_bytes(self) -> int:
        """The plain full-K/V-cache term, computed the same way
        `FullAttentionSpec.page_size_bytes` does (2x for K and V,
        `block_size` tokens, `num_kv_heads * head_size` elements per
        token). Not calling `super().page_size_bytes` directly since
        `AttentionSpec` itself does not implement it (only concrete
        subclasses like `FullAttentionSpec` do, and this class
        intentionally does not inherit from `FullAttentionSpec` to
        avoid implying it's fully interchangeable with one everywhere
        `FullAttentionSpec` is pattern-matched in vLLM internals --
        verified this distinction matters by checking
        `single_type_kv_cache_manager.py`'s `spec_manager_map`, which
        dispatches on exact type)."""
        elem_size = get_dtype_size(self.dtype)
        return 2 * self.block_size * self.num_kv_heads * self.head_size * elem_size

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        from vllm.utils.math_utils import cdiv

        num_blocks = cdiv(max_model_len, self.block_size)
        return num_blocks * self.page_size_bytes


class HierarchicalCompressedAttentionManager(SingleTypeKVCacheManager):
    """Stage 3b: cache-hit / allocation logic for
    `HierarchicalCompressedAttentionSpec`. See module docstring for the
    "why prefix-cache reuse is disabled" rationale (mirrors
    `MambaManager.get_num_common_prefix_blocks`'s real precedent
    exactly, same underlying reason).

    Everything NOT overridden here (basic block allocation/freeing,
    `get_num_blocks_to_allocate`, `cache_blocks`) falls through to
    `SingleTypeKVCacheManager`'s base implementation, which treats
    blocks generically via the block pool -- appropriate for this
    layer's dominant full-K/V-cache memory term (Phase 1 §2c), which
    behaves like any other append-only attention cache for allocation
    purposes. The compression-tier bookkeeping (which rows of
    `compress_k`/`compress_k2` exist, the ring-buffer staging state)
    lives in the model-side cache object at runtime (analogous to the
    reference `InfLLMv2CacheLayer`), not in this scheduler-side manager
    -- the manager's job is block-level memory accounting, not tracking
    the compression algorithm's internal state, exactly mirroring how
    `MambaManager` doesn't know or care about Mamba's internal SSM
    recurrence, only about block-level cache lifecycle.
    """

    def __init__(
        self,
        kv_cache_spec: HierarchicalCompressedAttentionSpec,
        block_pool: "BlockPool",
        **kwargs,
    ) -> None:
        assert isinstance(kv_cache_spec, HierarchicalCompressedAttentionSpec), (
            "HierarchicalCompressedAttentionManager can only be used for "
            "HierarchicalCompressedAttentionSpec groups"
        )
        super().__init__(kv_cache_spec, block_pool, **kwargs)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """Cascade attention / cross-request prefix sharing is NOT
        supported for this cache type -- see module docstring. Mirrors
        `MambaManager.get_num_common_prefix_blocks`'s real docstring
        verbatim in spirit ("cascade attention is not supported by
        mamba"): InfLLM-V2's incremental per-token compression state is
        sequentially stateful per-request, not safely shareable.
        """
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: "BlockHashList",
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: "BlockPool",
        kv_cache_spec: "HierarchicalCompressedAttentionSpec",
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list["KVCacheBlock"], ...]:
        """No cache hits, ever -- consistent with
        `get_num_common_prefix_blocks` returning 0 above. Returns the
        same "no blocks reused" shape `MambaManager.find_longest_cache_hit`
        returns when it finds no match (a tuple of empty lists, one per
        kv_cache_group_id), rather than raising, so this composes
        correctly with vLLM's generic multi-group prefix-cache-hit
        aggregation logic (which expects every group's manager to answer
        this call, not skip it).
        """
        assert isinstance(kv_cache_spec, HierarchicalCompressedAttentionSpec), (
            "HierarchicalCompressedAttentionManager can only be used for "
            "HierarchicalCompressedAttentionSpec groups"
        )
        assert dcp_world_size == 1, "DCP not supported for InfLLM-V2 sparse cache yet."
        assert pcp_world_size == 1, "PCP not supported for InfLLM-V2 sparse cache yet."
        return tuple([] for _ in range(len(kv_cache_group_ids)))


register_kv_cache_spec(
    manager_class=HierarchicalCompressedAttentionManager,
    uniform_type_base_spec=HierarchicalCompressedAttentionSpec,
)(HierarchicalCompressedAttentionSpec)
