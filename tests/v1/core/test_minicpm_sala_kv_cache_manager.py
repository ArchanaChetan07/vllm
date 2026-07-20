# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for `HierarchicalCompressedAttentionManager` (Stage 3b).

Previously flagged in docs/minicpm_sala_known_limitations.md as "has not
been imported, instantiated, or exercised at all" -- this file retires
that risk. Constructs a REAL `BlockPool` and a real manager instance,
exercising both required abstract methods
(`get_num_common_prefix_blocks`, `find_longest_cache_hit`) and both
`isinstance` guards. No GPU needed -- pure Python object construction
and method calls.
"""

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.minicpm_sala_kv_cache_spec import (
    HierarchicalCompressedAttentionManager,
    HierarchicalCompressedAttentionSpec,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _make_spec() -> HierarchicalCompressedAttentionSpec:
    return HierarchicalCompressedAttentionSpec(
        block_size=256,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        compress_kernel_size=32,
        compress_kernel_stride=16,
        dense_len=8192,
    )


def _make_manager(
    spec: HierarchicalCompressedAttentionSpec | None = None,
) -> HierarchicalCompressedAttentionManager:
    block_pool = BlockPool(num_gpu_blocks=100, enable_caching=True, hash_block_size=256)
    return HierarchicalCompressedAttentionManager(
        kv_cache_spec=spec or _make_spec(),
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=256,
    )


class TestConstruction:
    def test_real_construction_succeeds(self) -> None:
        manager = _make_manager()
        assert isinstance(manager, HierarchicalCompressedAttentionManager)

    def test_rejects_wrong_spec_type_at_construction(self) -> None:
        wrong_spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
        )
        block_pool = BlockPool(
            num_gpu_blocks=100, enable_caching=True, hash_block_size=16
        )
        import pytest

        with pytest.raises(AssertionError, match="HierarchicalCompressedAttentionSpec"):
            HierarchicalCompressedAttentionManager(
                kv_cache_spec=wrong_spec,
                block_pool=block_pool,
                enable_caching=True,
                kv_cache_group_id=0,
                scheduler_block_size=16,
            )


class TestGetNumCommonPrefixBlocks:
    def test_always_returns_zero(self) -> None:
        """Cascade attention / cross-request prefix sharing is
        deliberately disabled -- see the manager's own docstring for the
        full rationale (mirrors MambaManager's real precedent)."""
        manager = _make_manager()
        assert manager.get_num_common_prefix_blocks("req-1") == 0
        assert manager.get_num_common_prefix_blocks("any-other-id") == 0


class TestFindLongestCacheHit:
    def test_returns_correct_number_of_empty_lists(self) -> None:
        spec = _make_spec()
        block_pool = BlockPool(
            num_gpu_blocks=100, enable_caching=True, hash_block_size=256
        )
        for n_groups in (1, 3, 5):
            result = HierarchicalCompressedAttentionManager.find_longest_cache_hit(
                block_hashes=None,  # never touched by this implementation
                max_length=1000,
                kv_cache_group_ids=list(range(n_groups)),
                block_pool=block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=False,
                alignment_tokens=0,
            )
            assert len(result) == n_groups
            assert all(r == [] for r in result)

    def test_rejects_wrong_spec_type(self) -> None:
        import pytest

        wrong_spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
        )
        block_pool = BlockPool(
            num_gpu_blocks=100, enable_caching=True, hash_block_size=16
        )
        with pytest.raises(AssertionError, match="HierarchicalCompressedAttentionSpec"):
            HierarchicalCompressedAttentionManager.find_longest_cache_hit(
                block_hashes=None,
                max_length=1000,
                kv_cache_group_ids=[0],
                block_pool=block_pool,
                kv_cache_spec=wrong_spec,
                drop_eagle_block=False,
                alignment_tokens=0,
            )

    def test_rejects_dcp_world_size_other_than_one(self) -> None:
        import pytest

        spec = _make_spec()
        block_pool = BlockPool(
            num_gpu_blocks=100, enable_caching=True, hash_block_size=256
        )
        with pytest.raises(AssertionError, match="DCP not supported"):
            HierarchicalCompressedAttentionManager.find_longest_cache_hit(
                block_hashes=None,
                max_length=1000,
                kv_cache_group_ids=[0],
                block_pool=block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=False,
                alignment_tokens=0,
                dcp_world_size=2,
            )

    def test_rejects_pcp_world_size_other_than_one(self) -> None:
        import pytest

        spec = _make_spec()
        block_pool = BlockPool(
            num_gpu_blocks=100, enable_caching=True, hash_block_size=256
        )
        with pytest.raises(AssertionError, match="PCP not supported"):
            HierarchicalCompressedAttentionManager.find_longest_cache_hit(
                block_hashes=None,
                max_length=1000,
                kv_cache_group_ids=[0],
                block_pool=block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=False,
                alignment_tokens=0,
                pcp_world_size=2,
            )
