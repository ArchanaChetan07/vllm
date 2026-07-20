# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HierarchicalCompressedAttentionSpec (Stage 3a: the
KVCacheSpec for InfLLM-V2 sparse layers). No GPU needed -- pure
dataclass/arithmetic logic, actually run against a live vLLM install
during development (see docs/minicpm_sala_known_limitations.md).
"""

import torch

from vllm.v1.core.minicpm_sala_kv_cache_spec import (
    HierarchicalCompressedAttentionSpec,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

REAL_SPARSE_CONFIG = {
    "num_kv_heads": 2,
    "head_size": 128,
    "dtype": torch.bfloat16,
    "compress_kernel_size": 32,
    "compress_kernel_stride": 16,
    "dense_len": 8192,
}


class TestTierDerivation:
    def test_tier2_is_exactly_4x_tier1(self) -> None:
        spec = HierarchicalCompressedAttentionSpec(block_size=16, **REAL_SPARSE_CONFIG)
        assert spec.compress_k2_kernel_size == spec.compress_kernel_size * 4
        assert spec.compress_k2_kernel_stride == spec.compress_kernel_stride * 4
        assert spec.compress_k2_kernel_size == 128
        assert spec.compress_k2_kernel_stride == 64


class TestMemoryShape:
    """See Phase 1 report §2c / §-3 in known_limitations.md: the sparse
    layer's underlying attention IS a full K/V cache (no sub-linear
    savings there), and per the FINAL design (recompute compression
    tiers fresh each call rather than persist them -- see
    `minicpm_sala_sparse.py`'s `get_kv_cache_shape` design note),
    `page_size_bytes` now equals plain full attention's EXACTLY. This
    class name is kept even though "memory shape" is now the simplest
    possible answer (identical to FullAttentionSpec) -- worth asserting
    explicitly rather than assumed, since an earlier design iteration of
    this same file asserted the opposite (strictly larger) and that
    assertion was real and correct for THAT iteration.
    """

    def test_full_kv_component_matches_plain_full_attention_exactly(self) -> None:
        for block_size in (16, 64, 256):
            spec = HierarchicalCompressedAttentionSpec(
                block_size=block_size, **REAL_SPARSE_CONFIG
            )
            full_spec = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=REAL_SPARSE_CONFIG["num_kv_heads"],
                head_size=REAL_SPARSE_CONFIG["head_size"],
                dtype=REAL_SPARSE_CONFIG["dtype"],
            )
            assert spec._full_kv_page_size_bytes == full_spec.page_size_bytes

    def test_page_size_bytes_equals_plain_full_attention_exactly(self) -> None:
        """FINAL design: page_size_bytes == FullAttentionSpec's, at
        every block size, since compression tiers are never persisted
        in vLLM-managed cache memory (recomputed fresh from the full K
        cache on every call instead -- see minicpm_sala_sparse.py).
        This must hold exactly, not approximately: if it doesn't,
        vLLM's real memory planner (which allocates GPU memory based on
        `get_kv_cache_shape`, a physical allocation) and this spec's
        byte accounting (used for capacity planning) would disagree
        about how much memory this layer type actually needs -- a
        correctness bug in cache admission, not just a documentation
        mismatch.
        """
        for block_size in (16, 32, 64, 128, 256, 1024):
            spec = HierarchicalCompressedAttentionSpec(
                block_size=block_size, **REAL_SPARSE_CONFIG
            )
            full_spec = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=REAL_SPARSE_CONFIG["num_kv_heads"],
                head_size=REAL_SPARSE_CONFIG["head_size"],
                dtype=REAL_SPARSE_CONFIG["dtype"],
            )
            assert spec.page_size_bytes == full_spec.page_size_bytes, (
                f"at block_size={block_size}, sparse spec "
                f"({spec.page_size_bytes}) must exactly match plain "
                f"full attention ({full_spec.page_size_bytes}) -- "
                f"compression tiers are not persisted in cache memory "
                f"under the final (recompute-every-time) design"
            )

    def test_max_memory_usage_bytes_scales_with_model_len(self) -> None:
        import types

        spec = HierarchicalCompressedAttentionSpec(block_size=16, **REAL_SPARSE_CONFIG)
        small_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=4096)
        )
        large_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=16384)
        )
        small = spec.max_memory_usage_bytes(small_config)
        large = spec.max_memory_usage_bytes(large_config)
        assert large == small * 4, (
            "doubling max_model_len twice (4096->16384) should exactly "
            "quadruple total memory usage, since num_blocks scales "
            "linearly with max_model_len and page_size_bytes is fixed"
        )
