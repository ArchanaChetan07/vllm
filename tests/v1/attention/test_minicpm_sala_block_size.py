# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for page block_size propagation into gather (H1)."""

import pytest
import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    _assert_k_cache_page_size,
    _gather_full_k_with_new_tokens,
)


@pytest.mark.parametrize("block_size", [4, 16, 256])
def test_gather_respects_page_block_size(block_size: int) -> None:
    num_kv_heads, head_size = 1, 2
    num_physical_blocks = 8
    k_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_size)
    # Fill block 1 with token ids 0..block_size-1
    for t in range(block_size):
        k_cache[1, t, 0, 0] = float(t)

    n_cached = block_size
    block_table = torch.tensor([[1, 0, 0, 0]], dtype=torch.int32)
    new_key = torch.tensor([[99.0, 99.0]]).view(1, 1, 2)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    seq_lens_before = torch.tensor([n_cached], dtype=torch.int32)

    full_k, cu = _gather_full_k_with_new_tokens(
        k_cache=k_cache,
        new_key=new_key,
        block_table=block_table,
        seq_lens_before=seq_lens_before,
        query_start_loc=query_start_loc,
        block_size=block_size,
    )
    assert cu.tolist() == [0, n_cached + 1]
    ids = full_k[:, 0, 0].tolist()
    assert ids[:block_size] == [float(i) for i in range(block_size)]
    assert ids[-1] == 99.0


def test_assert_k_cache_page_size_mismatch_raises() -> None:
    k_cache = torch.zeros(2, 16, 1, 2)
    with pytest.raises(ValueError, match="KV page size mismatch"):
        _assert_k_cache_page_size(k_cache, page_block_size=256)
