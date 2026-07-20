# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttentionBackend/AttentionImpl for MiniCPM-SALA InfLLM-V2 sparse layers.

Kernel contracts are grounded in OpenBMB/infllmv2_cuda_impl. Page KV layout
matches FlashAttentionBackend: ``(num_blocks, 2, page_block_size, ...)``.
Sparse top-k scoring uses ``hf_config.sparse_config.block_size`` (typically 64),
which is distinct from the paged-attention page size (``cache_config.block_size``,
must be a multiple of 256 per infllm_v2).
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import nn

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

try:
    from infllm_v2 import (
        infllmv2_attn_stage1,
        infllmv2_attn_varlen_func,
        max_pooling_1d_varlen,
    )

    INFLLM_V2_AVAILABLE = True
except ImportError:
    # Mirrors the reference HF modeling file's own
    # `try: from infllm_v2 import ...; except ImportError: pass` pattern.
    # NOTE: `infllmv2_attn_varlen_func` (not `infllmv2_attn_with_kvcache`)
    # is the attention entry point here, matching the reference
    # `sparse_forward`. Verified against the installed kernel package on
    # A100 (2026-07-17): `with_kvcache` requires batched
    # (batch, seqlen_q, heads, head) input and cannot express vLLM's
    # packed varlen batches; `varlen_func` takes packed
    # (total_tokens, heads, head) q with cu_seqlens plus a paged-KV
    # `block_table` and an optional `topk_idx`.
    INFLLM_V2_AVAILABLE = False
    infllmv2_attn_varlen_func = None
    infllmv2_attn_stage1 = None
    max_pooling_1d_varlen = None


@dataclass(frozen=True)
class MiniCPMSALASparseConfig:
    """Runtime sparse-regime parameters from ``hf_config.sparse_config``.

    ``sparse_block_size`` is the top-k *scoring* block size (reference default 64).
    ``page_block_size`` is the paged KV page size from ``cache_config.block_size``
    (must be a multiple of 256 for infllm_v2); passed separately at construction.
    """

    kernel_size: int
    kernel_stride: int
    dense_len: int
    init_blocks: int
    topk: int
    window_size: int
    sparse_block_size: int

    @property
    def compress_k2_kernel_size(self) -> int:
        return self.kernel_size * 4

    @property
    def compress_k2_kernel_stride(self) -> int:
        return self.kernel_stride * 4

    @property
    def local_blocks(self) -> int:
        if self.window_size % self.sparse_block_size != 0:
            raise ValueError(
                f"sparse_config.window_size ({self.window_size}) must be "
                f"divisible by sparse_config.block_size ({self.sparse_block_size})"
            )
        return self.window_size // self.sparse_block_size

    @property
    def effective_topk(self) -> int:
        """Top-k actually passed to ``compressed_attention``.

        The reference ``MiniCPMInfLLMv2Attention.__init__`` sets
        ``self.topk = sparse_config["topk"] + window_size // block_size``
        (64 + 32 = 96 for the released checkpoint) -- the local-window
        blocks are budgeted ON TOP of the configured top-k, not carved out
        of it. Passing the raw config ``topk`` here would silently select
        32 fewer remote blocks than the reference."""
        return self.topk + self.local_blocks


def _sparse_config_field(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        if key not in raw:
            raise ValueError(f"sparse_config missing required field {key!r}")
        return raw[key]
    if not hasattr(raw, key):
        raise ValueError(f"sparse_config missing required attribute {key!r}")
    return getattr(raw, key)


def parse_sparse_config(hf_config: Any) -> MiniCPMSALASparseConfig:
    """Read and validate ``hf_config.sparse_config`` (no duplicated constants)."""
    raw = getattr(hf_config, "sparse_config", None)
    if raw is None:
        raise ValueError("MiniCPM-SALA requires hf_config.sparse_config; got None")
    cfg = MiniCPMSALASparseConfig(
        kernel_size=int(_sparse_config_field(raw, "kernel_size")),
        kernel_stride=int(_sparse_config_field(raw, "kernel_stride")),
        dense_len=int(_sparse_config_field(raw, "dense_len")),
        init_blocks=int(_sparse_config_field(raw, "init_blocks")),
        topk=int(_sparse_config_field(raw, "topk")),
        window_size=int(_sparse_config_field(raw, "window_size")),
        sparse_block_size=int(_sparse_config_field(raw, "block_size")),
    )
    if cfg.kernel_size <= 0 or cfg.kernel_stride <= 0:
        raise ValueError(
            f"sparse_config kernel_size/kernel_stride must be positive, "
            f"got {cfg.kernel_size}/{cfg.kernel_stride}"
        )
    if cfg.dense_len <= 0:
        raise ValueError(
            f"sparse_config.dense_len must be positive, got {cfg.dense_len}"
        )
    if cfg.topk <= 0:
        raise ValueError(f"sparse_config.topk must be positive, got {cfg.topk}")
    if cfg.sparse_block_size <= 0:
        raise ValueError(
            f"sparse_config.block_size must be positive, got {cfg.sparse_block_size}"
        )
    _ = cfg.local_blocks  # validates window_size divisibility
    return cfg


def validate_page_block_size(page_block_size: int) -> None:
    """The infllm_v2 paged-KV path requires page_block_size % 256 == 0."""
    if page_block_size <= 0:
        raise ValueError(f"page block_size must be positive, got {page_block_size}")
    if page_block_size % 256 != 0:
        raise ValueError(
            f"MiniCPM-SALA sparse page block_size must be a multiple of 256 "
            f"(infllm_v2 constraint), got {page_block_size}"
        )


# Sparse-regime boundary is inclusive: ``seq_len == dense_len`` selects sparse
# attention (dense applies only when ``seq_len < dense_len``). CONFIRMED
# against the current HF ``main`` snapshot of modeling_minicpm_sala.py
# (2026-07-17): ``MiniCPMInfLLMv2Attention.forward`` dispatches dense via
# ``if kv_seq_len < self.dense_len`` and sparse otherwise.
def sequence_sparse_mask(seq_lens: torch.Tensor, dense_len: int) -> torch.Tensor:
    """Per-sequence sparse-regime mask: True when ``seq_len >= dense_len``."""
    return seq_lens >= dense_len


def _assert_k_cache_page_size(k_cache: torch.Tensor, page_block_size: int) -> None:
    if k_cache.ndim != 4:
        raise ValueError(
            f"Expected k_cache shape (num_blocks, page_block_size, H, D), "
            f"got ndim={k_cache.ndim}"
        )
    if k_cache.shape[1] != page_block_size:
        raise ValueError(
            f"KV page size mismatch: k_cache page dim is {k_cache.shape[1]}, "
            f"expected page_block_size={page_block_size}. Check cache_config "
            f"propagation into Attention(..., block_size=...)."
        )


def calc_chunks_with_stride(
    cu_seqlen: torch.Tensor, chunk_size: int, kernel_stride: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Faithful, direct port of the reference `calc_chunks_with_stride`
    (modeling_minicpm_sala.py, fetched from the real source at commit
    9180fe1 -- copied line-for-line in logic, not reconstructed from
    the Phase 1 report's prose description of it). Computes the
    overlapping compression-window start offsets (stride=kernel_stride,
    width=chunk_size) within each packed sequence, and the resulting
    per-sequence compressed-row counts.

    NOTE: the reference decorates this with `@lru_cache(maxsize=16)`,
    keyed on the `cu_seqlen` tensor itself. Not reproduced here --
    `cu_seqlen` is a tensor (unhashable in the way `lru_cache` needs
    without `tensor.__hash__` support, which torch.Tensor does define
    but by identity, not value -- meaning the reference's caching only
    ever hits for the literal same tensor object, not equal-valued ones,
    a subtlety worth being aware of if reproducing the caching behavior
    is later judged worthwhile for performance; skipped here as a
    correctness-first-only concern per this project's own staging
    philosophy).
    """
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]

    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])

    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    chunk_indices = torch.arange(0, chunk_size, device=cu_seqlen.device)[None, :]
    filtered_indices = (valid_chunk_starts[:, None] + chunk_indices).view(-1)

    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    return filtered_indices, cu_seqlens_compressed


class CompressK(nn.Module):
    """Faithful, direct port of the reference `CompressK` module (same
    source as `calc_chunks_with_stride` above). Pure PyTorch, no
    `infllm_v2` dependency -- unlike the attention kernels themselves,
    this compression step has no custom CUDA kernel in the reference; it
    is plain `index_select` + `mean`, and is therefore fully portable and
    testable without any external package, GPU compilation step, or
    CUDA toolkit -- confirmed by reading the actual reference forward()
    body, which uses only `torch.Tensor.index_select`/`.view`/`.mean`.
    """

    def __init__(
        self, head_num_k: int, head_dim: int, kernel_size: int, kernel_stride: int = 16
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(
        self, k: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: (total_seq_len, num_heads, head_dim) -- packed/varlen keys,
                same layout vLLM's own attention metadata already uses
                (no reshaping needed at the call site beyond what any
                other varlen-format vLLM attention path already does).
            cu_seqlens: (batch_size + 1,) cumulative sequence lengths.
        Returns:
            compressed_k: (num_compressed_rows, num_heads, head_dim)
            cu_seqlens_compressed: (batch_size + 1,)
        """
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )
        compressed_k = filtered_k.mean(dim=1)
        return compressed_k, cu_seqlens_compressed


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Faithful, direct port of the reference `compressed_attention`
    function. Computes per-query-token top-k block indices over the
    compressed-key tiers.

    REAL DETAIL preserved exactly, not guessed independently (easy to
    get wrong without reading the actual source): `infllmv2_attn_stage1`
    is called with the tier-2 compressed keys `k2` passed as its `v`
    argument and `cu_seqlens_k2` passed as `cu_seqlens_v` -- i.e. the
    kernel's "value" input slot is being repurposed to carry the SECOND
    compression tier, not literal attention values. This is exactly what
    the reference does (`infllmv2_attn_stage1(q, k, k2, ...,
    cu_seqlens_v=cu_seqlens_k2, ...)`); reproduced verbatim here rather
    than "corrected" to look more conventional, since changing it would
    silently diverge from the real kernel contract.
    """
    if not INFLLM_V2_AVAILABLE:
        raise ImportError(
            "compressed_attention requires the infllm_v2 package "
            "(infllmv2_attn_stage1, max_pooling_1d_varlen) -- see "
            "MiniCPMSALASparseAttentionImpl's __init__ for the same "
            "check and its rationale."
        )
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1
        is_prefilling = cache_lens is None or bool((cache_lens == 0).all().item())

        if is_prefilling:
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
            q_idx = torch.cat(
                [
                    (
                        torch.arange(
                            cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device
                        )
                        + max_seqlen_q
                        - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                    )
                    // block_size
                    for i in range(batch_size)
                ],
                dim=0,
            )
        else:
            # Decode (one token per sequence) or CHUNKED PREFILL (several
            # tokens per sequence on top of a non-empty cache). vLLM v1
            # chunks long prefills; the HF reference never sees that case
            # -- its decode formula `cache_lens // block_size` assumes
            # exactly one query token per sequence and crashes the
            # max_pooling kernel's total_q check otherwise. Generalize per
            # token: absolute position (cache_len + local index) //
            # block_size. For q_len == 1 this reduces exactly to the
            # reference decode formula.
            q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            q_idx = (
                torch.cat(
                    [
                        cache_lens[i] + torch.arange(int(q_lens[i]), device=q.device)
                        for i in range(batch_size)
                    ]
                )
                // block_size
            )

        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            k2.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k2,  # k2 rides the "v" slot, see docstring
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_prefilling,
        )
        score = score[:, : q_idx.shape[0], :]

        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )

        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


@dataclass
class MiniCPMSALASparseAttentionMetadata:
    """Per-layer sparse-attention metadata."""

    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    dense_len: int
    page_block_size: int
    # Physical cache slot per new token (CommonAttentionMetadata.slot_mapping).
    # The impl writes new K/V into the paged cache via reshape_and_cache_flash
    # before attending. None => the caller has already populated the cache
    # (sub-batches inside _forward_mixed; some unit tests).
    slot_mapping: torch.Tensor | None = None


class MiniCPMSALASparseAttentionMetadataBuilder(
    AttentionMetadataBuilder[MiniCPMSALASparseAttentionMetadata]
):
    """Translates vLLM's common metadata into sparse-layer field names."""

    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig | None",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniCPMSALASparseAttentionMetadata:
        dense_len = getattr(self.kv_cache_spec, "dense_len", None)
        if dense_len is None:
            raise ValueError(
                "HierarchicalCompressedAttentionSpec (or compatible spec) "
                "with dense_len is required for MiniCPM-SALA sparse layers"
            )
        page_block_size = self.kv_cache_spec.block_size
        validate_page_block_size(page_block_size)
        return MiniCPMSALASparseAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            dense_len=int(dense_len),
            page_block_size=int(page_block_size),
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def update_block_table(
        self,
        metadata: MiniCPMSALASparseAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> MiniCPMSALASparseAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class MiniCPMSALASparseAttentionBackend(AttentionBackend):
    """Real signatures throughout -- see module docstring for the source
    this was grounded against."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    # From the real infllm_v2 paged-KV contract: "page_block_size
    # must be a multiple of 256" -- NOT the same constraint as
    # FlashAttentionBackend's `MultipleOf(16)`, deliberately not copied
    # from there.
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(256)]

    @staticmethod
    def get_name() -> str:
        # "CUSTOM" is vLLM's sanctioned name for out-of-tree backends:
        # AttentionBackendEnum resolves names strictly, and third-party
        # backends register their class under AttentionBackendEnum.CUSTOM
        # (see the register_backend call at the bottom of this module).
        # A bespoke name here makes engine init fail with
        # "Unknown attention backend: 'MINICPM_SALA_INFLLM_V2'".
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        # Decoder-only causal attention -- this model has no
        # encoder/cross-attention layers (Phase 1 report: text-only
        # causal LM).
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["MiniCPMSALASparseAttentionImpl"]:
        return MiniCPMSALASparseAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MiniCPMSALASparseAttentionMetadataBuilder"]:
        return MiniCPMSALASparseAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 256 != 0:
            raise ValueError(
                "MiniCPM-SALA sparse attention block_size must be a "
                "multiple of 256 (real constraint from infllm_v2's "
                "infllm_v2 paged-KV contract: 'page_block_size "
                "must be a multiple of 256'), got "
                f"block_size={block_size}."
            )
        # REVISED (previously reserved extra space for persistent
        # tier1/tier2 compressed-K storage; reverted back to full-K/V
        # only). Real design decision made while actually wiring up the
        # sparse forward path (see `_forward_sparse` below): rather than
        # persist compressed tier rows across decode steps (which would
        # need real incremental-update bookkeeping -- exactly the class
        # of stateful logic flagged as too risky to write blind since
        # Stage 3b), compression tiers are recomputed FRESH on every
        # call from the full K cache. This is the "measure before
        # optimizing" tradeoff already flagged in
        # `HierarchicalCompressedAttentionSpec`'s design note -- shipping
        # the simpler, more obviously-correct version first. Persistent
        # tier caching (requiring this shape to grow again, symmetric
        # with `HierarchicalCompressedAttentionSpec.page_size_bytes`,
        # which was ALSO reverted to full-KV-only for the same reason --
        # see that file's own updated design note) is legitimate future
        # work once real profiling shows the recompute cost matters, not
        # before.
        #
        # (num_blocks, 2, block_size, num_kv_heads, head_size) -- SAME
        # convention as FlashAttentionBackend.get_kv_cache_shape (the
        # "2" packs K and V into one tensor; confirmed by reading
        # vllm/v1/attention/backends/flash_attn.py directly).
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


class MiniCPMSALASparseAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        *,
        block_size: int,
        sparse_config: MiniCPMSALASparseConfig,
    ) -> None:
        if not INFLLM_V2_AVAILABLE:
            raise ImportError(
                "MiniCPMSALASparseAttentionImpl requires the infllm_v2 package "
                "(OpenBMB/infllmv2_cuda_impl), which is not installed."
            )
        assert attn_type == AttentionType.DECODER
        validate_page_block_size(block_size)
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.page_block_size = block_size
        self.sparse_config = sparse_config
        self.compress_k1 = CompressK(
            head_num_k=self.num_kv_heads,
            head_dim=self.head_size,
            kernel_size=sparse_config.kernel_size,
            kernel_stride=sparse_config.kernel_stride,
        )
        self.compress_k2 = CompressK(
            head_num_k=self.num_kv_heads,
            head_dim=self.head_size,
            kernel_size=sparse_config.compress_k2_kernel_size,
            kernel_stride=sparse_config.compress_k2_kernel_stride,
        )
        assert sliding_window is None
        assert alibi_slopes is None
        assert logits_soft_cap is None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            # Profiling run (same convention as FlashAttentionImpl):
            # kv_cache is a dummy 1-D placeholder and there is nothing to
            # attend over.
            return output

        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
        # The metadata/cache page size is authoritative, NOT the
        # construction-time cache_config.block_size: vLLM's hybrid KV-cache
        # unification pads the attention block_size so every cache group
        # (incl. the lightning layers' recurrent-state pages) shares one
        # page byte-size -- e.g. 256 -> 2048 for this model on A100. Any
        # multiple of 256 satisfies the infllm_v2 kernel constraint.
        page_block_size = getattr(
            attn_metadata, "page_block_size", self.page_block_size
        )
        validate_page_block_size(page_block_size)
        _assert_k_cache_page_size(k_cache, page_block_size)

        # Write the new K/V tokens into the paged cache first -- same
        # convention as FlashAttentionImpl.forward. The attention calls
        # below then read K/V exclusively through (k_cache, v_cache,
        # block_table), which keeps prefill, decode, and mixed batches on
        # one uniform paged-varlen path.
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is not None:
            from vllm.v1.attention.backends.fa_utils import (
                reshape_and_cache_flash,
            )

            k_scale = getattr(layer, "_k_scale", None) if layer is not None else None
            v_scale = getattr(layer, "_v_scale", None) if layer is not None else None
            if k_scale is None:
                # Standalone/validation-script use (layer=None): unquantized
                # cache, unit scales.
                k_scale = torch.ones(1, dtype=torch.float32, device=query.device)
                v_scale = k_scale
            reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                self.kv_cache_dtype,
                k_scale,
                v_scale,
            )

        dense_len = attn_metadata.dense_len
        sparse_mask = sequence_sparse_mask(attn_metadata.seq_lens, dense_len)
        if not sparse_mask.any():
            return self._forward_dense(
                query, key, value, k_cache, v_cache, attn_metadata, output
            )
        if sparse_mask.all():
            return self._forward_sparse(
                query, key, value, k_cache, v_cache, attn_metadata, output
            )
        return self._forward_mixed(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata,
            output,
            sparse_mask,
        )

    def _attend_varlen(
        self,
        query: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        attn_metadata,
        topk_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        """One contiguous-varlen attention call for prefill AND decode.

        `infllmv2_attn_varlen_func` (the reference `sparse_forward` entry
        point) takes packed (total_q, heads, head) queries and contiguous
        (total_k, kv_heads, head) keys/values -- the SAME layout the HF
        reference feeds it after `_upad_input`. Verified on the installed
        kernel package (A100, 2026-07-17): the fork's `topk_idx`
        preprocessing derives `nheads_k = k.shape[1]`, i.e. it requires
        contiguous 3D K and cannot take the paged cache directly, so K/V
        are gathered from the paged cache first (`_gather_full_k_with_new_
        tokens`). `causal=True` uses flash-attn's bottom-right-aligned
        mask, which is exactly full-row attention for decode rows
        (seqlen_q==1), so no decode special-case is needed. `topk_idx=None`
        is the dense mode; non-None selects the InfLLM-V2 sparse blocks.
        """
        qsl = attn_metadata.query_start_loc.to(torch.int32)
        q_lens = qsl[1:] - qsl[:-1]
        return infllmv2_attn_varlen_func(
            query,
            full_k,
            full_v,
            cu_seqlens_q=qsl,
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=int(q_lens.max().item()),
            max_seqlen_k=int(attn_metadata.seq_lens.max().item()),
            softmax_scale=self.scale,
            causal=True,
            topk_idx=topk_idx,
        )

    def _gather_full_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Contiguous (total_k, kv_heads, head) K and V: cached + new."""
        num_new_tokens = _num_new_tokens_per_seq(attn_metadata)
        seq_lens_before = attn_metadata.seq_lens - num_new_tokens
        # Metadata page size, not construction-time cache_config.block_size
        # -- see the hybrid page-size unification note in forward().
        block_size = getattr(attn_metadata, "page_block_size", self.page_block_size)
        full_k, cu_seqlens = _gather_full_k_with_new_tokens(
            k_cache=k_cache,
            new_key=key,
            block_table=attn_metadata.block_table,
            seq_lens_before=seq_lens_before,
            query_start_loc=attn_metadata.query_start_loc,
            block_size=block_size,
        )
        full_v, _ = _gather_full_k_with_new_tokens(
            k_cache=v_cache,
            new_key=value,
            block_table=attn_metadata.block_table,
            seq_lens_before=seq_lens_before,
            query_start_loc=attn_metadata.query_start_loc,
            block_size=block_size,
        )
        return full_k, full_v, cu_seqlens

    def _forward_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        full_k, full_v, cu_seqlens_k = self._gather_full_kv(
            key, value, k_cache, v_cache, attn_metadata
        )
        out = self._attend_varlen(
            query, full_k, full_v, cu_seqlens_k, attn_metadata, topk_idx=None
        )
        output.copy_(out.view(output.shape))
        return output

    def _forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        sc = self.sparse_config
        num_new_tokens = _num_new_tokens_per_seq(attn_metadata)
        full_k, full_v, cu_seqlens_full = self._gather_full_kv(
            key, value, k_cache, v_cache, attn_metadata
        )
        compressed_k, cu_seqlens_k1 = self.compress_k1(full_k, cu_seqlens_full)
        compressed_k2, cu_seqlens_k2 = self.compress_k2(full_k, cu_seqlens_full)

        # The infllm_v2 sparse kernels require a 16:1 q:kv head ratio.
        # Reference (`sparse_forward`): when the ratio is below 16, q heads
        # are repeat_interleaved up to 16 and the outputs of the copies are
        # averaged back. At TP=1/2 this model is exactly 16:1 (no-op);
        # under TP=4 each rank holds 8 q heads per replicated kv head
        # (ratio 8) and skipping the repeat crashes stage1/max_pooling.
        num_q_heads = query.shape[1]
        num_k_heads = full_k.shape[1]
        current_ratio = num_q_heads // num_k_heads
        repeat_times = 16 // current_ratio if current_ratio < 16 else 1
        q_kernels = (
            query.repeat_interleave(repeat_times, dim=1) if repeat_times > 1 else query
        )

        q_lens = attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        cache_lens = (attn_metadata.seq_lens - num_new_tokens).to(torch.int32)
        topk_idx = compressed_attention(
            q=q_kernels,
            k=compressed_k,
            k2=compressed_k2,
            kernel_size=sc.kernel_size,
            kernel_stride=sc.kernel_stride,
            block_size=sc.sparse_block_size,
            topk=sc.effective_topk,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=cu_seqlens_k1,
            cu_seqlens_k2=cu_seqlens_k2,
            max_seqlen_q=int(q_lens.max().item()),
            max_seqlen_k=int(cu_seqlens_k1[1:].max().item()),
            init_blocks=sc.init_blocks,
            local_blocks=sc.local_blocks,
            cache_lens=cache_lens,
        )

        out = self._attend_varlen(
            q_kernels, full_k, full_v, cu_seqlens_full, attn_metadata, topk_idx=topk_idx
        )
        if repeat_times > 1:
            # Average the repeated copies back to the true head count --
            # repeat_interleave put a head's copies adjacent, so grouping
            # adjacent `repeat_times` outputs matches the reference's
            # `.view(..., heads, repeat, -1).mean(dim=-2)`.
            out = out.view(out.shape[0], num_q_heads, repeat_times, out.shape[-1]).mean(
                dim=2
            )
        output.copy_(out.view(output.shape))
        return output

    def _forward_mixed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sequence dense vs sparse dispatch for mixed-length batches."""
        dense_indices = (~sparse_mask).nonzero(as_tuple=False).flatten().tolist()
        sparse_indices = sparse_mask.nonzero(as_tuple=False).flatten().tolist()

        if dense_indices:
            sub_q, sub_k, sub_v, sub_meta, ranges = _select_varlen_sequences(
                dense_indices, query, key, value, attn_metadata
            )
            sub_out = torch.empty_like(sub_q)
            self._forward_dense(
                sub_q, sub_k, sub_v, k_cache, v_cache, sub_meta, sub_out
            )
            # per-sequence scatter-back: sub_out is packed in ranges order
            sub_offset = 0
            for start, end in ranges:
                n = end - start
                output[start:end].copy_(sub_out[sub_offset : sub_offset + n])
                sub_offset += n
            assert sub_offset == sub_out.shape[0]

        if sparse_indices:
            sub_q, sub_k, sub_v, sub_meta, ranges = _select_varlen_sequences(
                sparse_indices, query, key, value, attn_metadata
            )
            sub_out = torch.empty_like(sub_q)
            self._forward_sparse(
                sub_q, sub_k, sub_v, k_cache, v_cache, sub_meta, sub_out
            )
            # per-sequence scatter-back: sub_out is packed in ranges order
            sub_offset = 0
            for start, end in ranges:
                n = end - start
                output[start:end].copy_(sub_out[sub_offset : sub_offset + n])
                sub_offset += n
            assert sub_offset == sub_out.shape[0]

        return output


def _num_new_tokens_per_seq(attn_metadata) -> torch.Tensor:
    return attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]


def _select_varlen_sequences(
    seq_indices: list[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: MiniCPMSALASparseAttentionMetadata,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MiniCPMSALASparseAttentionMetadata,
    list[tuple[int, int]],
]:
    """Extract a subset of sequences from a packed varlen batch."""
    if not seq_indices:
        raise ValueError("seq_indices must be non-empty")
    qsl = attn_metadata.query_start_loc.tolist()
    token_ranges: list[tuple[int, int]] = []
    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for i in seq_indices:
        start, end = qsl[i], qsl[i + 1]
        token_ranges.append((start, end))
        q_parts.append(query[start:end])
        k_parts.append(key[start:end])
        v_parts.append(value[start:end])

    sub_q = torch.cat(q_parts, dim=0)
    sub_k = torch.cat(k_parts, dim=0)
    sub_v = torch.cat(v_parts, dim=0)
    token_counts = [end - start for start, end in token_ranges]
    new_qsl = torch.zeros(len(seq_indices) + 1, dtype=torch.int32, device=query.device)
    new_qsl[1:] = torch.tensor(token_counts, dtype=torch.int32, device=query.device)
    new_qsl[1:] = new_qsl[1:].cumsum(dim=0)

    idx = torch.tensor(seq_indices, dtype=torch.long, device=query.device)
    sub_metadata = MiniCPMSALASparseAttentionMetadata(
        query_start_loc=new_qsl,
        seq_lens=attn_metadata.seq_lens.index_select(0, idx),
        block_table=attn_metadata.block_table.index_select(0, idx),
        dense_len=attn_metadata.dense_len,
        page_block_size=attn_metadata.page_block_size,
    )
    return sub_q, sub_k, sub_v, sub_metadata, token_ranges


def _gather_full_k_with_new_tokens(
    k_cache: torch.Tensor,
    new_key: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_before: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather cached + new K tokens into one contiguous varlen tensor."""
    _assert_k_cache_page_size(k_cache, block_size)
    num_seqs = seq_lens_before.shape[0]
    seq_lens_before_list = seq_lens_before.tolist()
    query_start_loc_list = query_start_loc.tolist()
    gathered_per_seq: list[torch.Tensor] = []
    for i in range(num_seqs):
        n_before = seq_lens_before_list[i]
        num_blocks_before = (n_before + block_size - 1) // block_size
        if num_blocks_before > 0:
            physical_blocks = block_table[i, :num_blocks_before]
            cached_k = k_cache[physical_blocks].reshape(
                num_blocks_before * block_size, *k_cache.shape[2:]
            )[:n_before]
        else:
            cached_k = k_cache.new_zeros((0, *k_cache.shape[2:]))

        new_start = query_start_loc_list[i]
        new_end = query_start_loc_list[i + 1]
        new_k_this_seq = new_key[new_start:new_end]
        gathered_per_seq.append(torch.cat([cached_k, new_k_this_seq], dim=0))

    full_k = torch.cat(gathered_per_seq, dim=0)
    seq_lens_after = [g.shape[0] for g in gathered_per_seq]
    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=full_k.device)
    cu_seqlens[1:] = torch.tensor(
        seq_lens_after, dtype=torch.int32, device=full_k.device
    ).cumsum(0)
    return full_k, cu_seqlens


# Register as vLLM's out-of-tree CUSTOM backend so AttentionBackendEnum can
# resolve this class by name (vLLM >= 0.25 validates backend names against
# the enum; get_name() above returns "CUSTOM" to match). Guarded for the
# 0.24 pin, whose registry predates register_backend/CUSTOM.
try:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(AttentionBackendEnum.CUSTOM)(MiniCPMSALASparseAttentionBackend)
except (ImportError, AttributeError, KeyError):
    pass
