# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for per-sequence sparse/dense dispatch helpers (H2)."""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseAttentionMetadata,
    _select_varlen_sequences,
    sequence_sparse_mask,
)


def _metadata(
    seq_lens: list[int],
    q_tokens_per_seq: list[int],
    page_block_size: int = 256,
    dense_len: int = 8192,
) -> MiniCPMSALASparseAttentionMetadata:
    qsl = [0]
    for n in q_tokens_per_seq:
        qsl.append(qsl[-1] + n)
    return MiniCPMSALASparseAttentionMetadata(
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.zeros(len(seq_lens), 4, dtype=torch.int32),
        dense_len=dense_len,
        page_block_size=page_block_size,
    )


class TestSequenceSparseMask:
    def test_all_dense(self) -> None:
        seq_lens = torch.tensor([100, 4096, 8191], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [False, False, False]

    def test_all_sparse(self) -> None:
        seq_lens = torch.tensor([8192, 9000, 100000], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [True, True, True]

    def test_mixed_batch(self) -> None:
        seq_lens = torch.tensor([4096, 8192, 8193, 0], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [False, True, True, False]

    def test_boundary_exact_dense_len(self) -> None:
        seq_lens = torch.tensor([8191, 8192], dtype=torch.int32)
        mask = sequence_sparse_mask(seq_lens, dense_len=8192)
        assert mask.tolist() == [False, True]


class TestSelectVarlenSequences:
    def test_subset_preserves_token_order(self) -> None:
        meta = _metadata(seq_lens=[100, 9000], q_tokens_per_seq=[2, 3])
        total = 5
        query = torch.arange(total).view(total, 1, 1).expand(total, 1, 2).float()
        key = query.clone()
        value = query.clone()

        sub_q, sub_k, sub_v, sub_meta, ranges = _select_varlen_sequences(
            [1], query, key, value, meta
        )
        assert sub_q.shape[0] == 3
        assert ranges == [(2, 5)]
        assert sub_meta.seq_lens.tolist() == [9000]
        assert sub_meta.query_start_loc.tolist() == [0, 3]
        assert sub_q[:, 0, 0].tolist() == [2.0, 3.0, 4.0]

    def test_empty_new_sequence_q_tokens(self) -> None:
        meta = _metadata(seq_lens=[0], q_tokens_per_seq=[0])
        query = torch.zeros(0, 1, 2)
        key = query.clone()
        value = query.clone()
        sub_q, _, _, sub_meta, ranges = _select_varlen_sequences(
            [0], query, key, value, meta
        )
        assert sub_q.shape[0] == 0
        assert ranges == [(0, 0)]
        assert sub_meta.seq_lens.tolist() == [0]
