# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ported CompressK / calc_chunks_with_stride
(vllm/v1/attention/backends/minicpm_sala_sparse.py). Both are pure
PyTorch with no `infllm_v2` dependency, so these tests need no CUDA
toolkit or external package -- but per explicit instruction, this file
has been written and statically checked (py_compile, ruff) only, NOT
executed, in this pass. Running it is real, low-risk, concrete next
work (torch-only, no GPU needed) -- flagged in
docs/minicpm_sala_known_limitations.md rather than silently left
undifferentiated from the rest of this file's untested status.
"""

import torch

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    CompressK,
    calc_chunks_with_stride,
)


class TestCalcChunksWithStride:
    def test_single_sequence_exact_multiple(self) -> None:
        # One sequence of 64 tokens, kernel_size=32, stride=16 -- the
        # real released checkpoint's tier-1 values (Phase 1 report).
        # Expected compressed rows: windows [0:32], [16:48], [32:64] = 3
        # (window starting at 48 would need [48:80], exceeds seq_len=64).
        cu_seqlens = torch.tensor([0, 64], dtype=torch.int32)
        filtered_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, chunk_size=32, kernel_stride=16
        )
        assert cu_seqlens_compressed.tolist() == [0, 3]
        assert filtered_indices.numel() == 3 * 32

    def test_two_sequences_independent_windows(self) -> None:
        # Sequence 0: 64 tokens (3 windows, as above). Sequence 1: 32
        # tokens exactly (1 window: [0:32]).
        cu_seqlens = torch.tensor([0, 64, 96], dtype=torch.int32)
        filtered_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, chunk_size=32, kernel_stride=16
        )
        assert cu_seqlens_compressed.tolist() == [0, 3, 4]
        assert filtered_indices.numel() == 4 * 32

    def test_sequence_shorter_than_kernel_size_produces_no_windows(self) -> None:
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)
        _, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, chunk_size=32, kernel_stride=16
        )
        assert cu_seqlens_compressed.tolist() == [0, 0]

    def test_tier2_parameters_real_checkpoint_values(self) -> None:
        # Tier-2 is always 4x tier-1 (kernel_size=128, stride=64) per
        # the reference `CompressK(..., kernel_size=self.kernel_size*4,
        # kernel_stride=self.kernel_stride*4)`. 256 tokens -> windows
        # [0:128], [64:192], [128:256] = 3.
        cu_seqlens = torch.tensor([0, 256], dtype=torch.int32)
        _, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, chunk_size=128, kernel_stride=64
        )
        assert cu_seqlens_compressed.tolist() == [0, 3]


class TestCompressK:
    def test_output_shape_and_mean_pooling_correctness(self) -> None:
        # Real checkpoint tier-1 params: num_kv_heads=2, head_dim=128,
        # kernel_size=32, kernel_stride=16 (Phase 1 report config
        # ground truth).
        compress_k = CompressK(
            head_num_k=2, head_dim=128, kernel_size=32, kernel_stride=16
        )
        k = torch.randn(64, 2, 128)
        cu_seqlens = torch.tensor([0, 64], dtype=torch.int32)
        compressed_k, cu_seqlens_compressed = compress_k(k, cu_seqlens)
        assert compressed_k.shape == (3, 2, 128)
        assert cu_seqlens_compressed.tolist() == [0, 3]

    def test_mean_pooling_matches_manual_computation(self) -> None:
        # Deterministic input (arange) so the mean-pool result can be
        # checked exactly, not just shape-checked.
        compress_k = CompressK(head_num_k=1, head_dim=1, kernel_size=4, kernel_stride=2)
        k = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
        cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
        compressed_k, cu_seqlens_compressed = compress_k(k, cu_seqlens)
        # Windows: [0,1,2,3]->mean=1.5, [2,3,4,5]->mean=3.5,
        # [4,5,6,7]->mean=5.5
        expected = torch.tensor([1.5, 3.5, 5.5]).view(3, 1, 1)
        assert torch.allclose(compressed_k, expected)
        assert cu_seqlens_compressed.tolist() == [0, 3]
