# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for `_gather_full_k_with_new_tokens` -- flagged in its own
docstring as the single highest-risk function in this project (real
paged-cache block_table indexing, never previously executed even once).

This test constructs a small, hand-verifiable synthetic paged cache --
deliberately non-contiguous physical block ordering and a partial final
block, to actually exercise the indirection rather than a trivial
identity-mapped case that would pass even with a buggy gather. Pure
CPU tensor indexing; no GPU needed.
"""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    _gather_full_k_with_new_tokens,
)


def test_gather_reconstructs_correct_token_order_across_noncontiguous_blocks() -> None:
    block_size = 4
    num_kv_heads, head_size = 1, 2
    num_physical_blocks = 6

    # Each token's stored value = its global token id, for easy
    # verification. Physical block layout is deliberately
    # non-contiguous and non-identity (sequence 0's logical blocks
    # [0, 1] map to physical blocks [5, 2], not [0, 1]) to actually
    # exercise block_table indirection, not just pass by coincidence.
    k_cache = torch.zeros(num_physical_blocks, block_size, num_kv_heads, head_size)
    for local_tok, tok_id in enumerate(range(0, 4)):
        k_cache[5, local_tok] = tok_id  # seq 0, logical block 0
    for local_tok, tok_id in enumerate(range(4, 6)):
        k_cache[2, local_tok] = (
            tok_id  # seq 0, logical block 1 (PARTIAL: only 2/4 slots)
        )
    for local_tok, tok_id in enumerate([100, 101, 102]):
        k_cache[0, local_tok] = (
            tok_id  # seq 1, logical block 0 (PARTIAL: only 3/4 slots)
        )

    block_table = torch.tensor(
        [
            [5, 2, 0, 0],  # seq 0's logical blocks -> physical [5, 2, pad, pad]
            [0, 0, 0, 0],  # seq 1's logical blocks -> physical [0, pad, pad, pad]
        ],
        dtype=torch.int32,
    )

    # New tokens this call: seq 0 gets 2 new tokens (ids 6, 7), seq 1
    # gets 1 new token (id 103) -- e.g. a chunked-prefill or decode step
    # continuing both sequences.
    new_key = torch.tensor([[6.0, 6.0], [7.0, 7.0], [103.0, 103.0]]).view(3, 1, 2)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    seq_lens_before = torch.tensor([6, 3], dtype=torch.int32)

    full_k, cu_seqlens = _gather_full_k_with_new_tokens(
        k_cache=k_cache,
        new_key=new_key,
        block_table=block_table,
        seq_lens_before=seq_lens_before,
        query_start_loc=query_start_loc,
        block_size=block_size,
    )

    assert cu_seqlens.tolist() == [0, 8, 12]
    assert full_k.shape == (12, num_kv_heads, head_size)
    actual_token_order = full_k[:, 0, 0].tolist()
    expected_token_order = [0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103]
    assert actual_token_order == expected_token_order, (
        f"expected {expected_token_order}, got {actual_token_order} -- "
        f"gather did not correctly reconstruct token order across "
        f"non-contiguous physical blocks and/or partial final blocks"
    )


def test_gather_handles_zero_cached_tokens() -> None:
    """First-ever call for a brand-new sequence: seq_lens_before=0, no
    cached tokens at all yet, only new tokens. Edge case worth its own
    test since `_gather_full_k_with_new_tokens`'s `num_blocks_before > 0`
    branch exists specifically for this."""
    block_size = 4
    k_cache = torch.zeros(2, block_size, 1, 2)
    block_table = torch.tensor([[0, 0]], dtype=torch.int32)
    new_key = torch.tensor([[1.0, 1.0], [2.0, 2.0]]).view(2, 1, 2)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens_before = torch.tensor([0], dtype=torch.int32)

    full_k, cu_seqlens = _gather_full_k_with_new_tokens(
        k_cache=k_cache,
        new_key=new_key,
        block_table=block_table,
        seq_lens_before=seq_lens_before,
        query_start_loc=query_start_loc,
        block_size=block_size,
    )
    assert cu_seqlens.tolist() == [0, 2]
    assert full_k.shape == (2, 1, 2)
    assert full_k[:, 0, 0].tolist() == [1.0, 2.0]
