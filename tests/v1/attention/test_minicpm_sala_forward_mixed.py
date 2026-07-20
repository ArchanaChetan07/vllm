# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for ``_forward_mixed`` per-token scatter-back."""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseAttentionImpl,
    MiniCPMSALASparseAttentionMetadata,
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


def _token_signature_query(total_tokens: int, num_heads: int = 1, head_dim: int = 2):
    """Packed query where row *i* encodes global token index *i* on every axis."""
    idx = torch.arange(total_tokens, dtype=torch.float32)
    return (
        idx.view(total_tokens, 1, 1).expand(total_tokens, num_heads, head_dim).clone()
    )


def _signature_forward(sub_q, _sub_k, _sub_v, _k_cache, _v_cache, _sub_meta, sub_out):
    """Fake dense/sparse path: copy per-token signature rows into sub_out."""
    sub_out.copy_(sub_q)


def _make_impl() -> MiniCPMSALASparseAttentionImpl:
    impl = object.__new__(MiniCPMSALASparseAttentionImpl)
    impl._forward_dense = _signature_forward
    impl._forward_sparse = _signature_forward
    return impl


def _run_forward_mixed(
    seq_lens: list[int],
    q_tokens_per_seq: list[int],
    dense_len: int = 8192,
) -> torch.Tensor:
    total = sum(q_tokens_per_seq)
    meta = _metadata(seq_lens, q_tokens_per_seq, dense_len=dense_len)
    query = _token_signature_query(total)
    key = query.clone()
    value = query.clone()
    output = torch.full_like(query, -1.0)
    k_cache = torch.zeros(1, 256, 1, 2)
    v_cache = torch.zeros(1, 256, 1, 2)
    sparse_mask = sequence_sparse_mask(meta.seq_lens, dense_len)

    impl = _make_impl()
    impl._forward_mixed(query, key, value, k_cache, v_cache, meta, output, sparse_mask)
    return output


class TestForwardMixedScatterBack:
    def test_mixed_batch_preserves_per_token_output(self) -> None:
        """Multi-token dense + sparse sequences must not smear the first token."""
        output = _run_forward_mixed(seq_lens=[100, 9000], q_tokens_per_seq=[3, 4])
        expected = torch.arange(output.shape[0], dtype=torch.float32)
        assert output[:, 0, 0].tolist() == expected.tolist()

    def test_pure_decode_still_correct(self) -> None:
        """One query token per sequence: old and new scatter paths agree."""
        output = _run_forward_mixed(seq_lens=[100, 9000], q_tokens_per_seq=[1, 1])
        expected = torch.arange(output.shape[0], dtype=torch.float32)
        assert output[:, 0, 0].tolist() == expected.tolist()
