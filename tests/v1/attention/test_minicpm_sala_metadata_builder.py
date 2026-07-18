# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPMSALASparseAttentionMetadataBuilder (Stage 4 gap
closed this round). Real bug this closes: every call site in
minicpm_sala_sparse.py referenced `attn_metadata.block_table`, but the
real `CommonAttentionMetadata` field is `block_table_tensor` (confirmed
via `dataclasses.fields`) -- the earlier unverified reuse of
FlashAttentionMetadataBuilder would have silently produced an
AttributeError the first time any of those call sites actually ran.
This builder fixes the mapping at the translation boundary.

No GPU needed -- pure Python object construction and method calls,
using a real (if minimal) CommonAttentionMetadata instance.
"""

import torch

from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseAttentionMetadata,
    MiniCPMSALASparseAttentionMetadataBuilder,
)
from vllm.v1.core.minicpm_sala_kv_cache_spec import HierarchicalCompressedAttentionSpec


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


def _make_common_attn_metadata(
    seq_len: int = 8, block_table: torch.Tensor | None = None
) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=(
            block_table
            if block_table is not None
            else torch.tensor([[0, 1, 2]], dtype=torch.int32)
        ),
        slot_mapping=torch.arange(seq_len, dtype=torch.int32),
    )


def _make_builder(
    spec: HierarchicalCompressedAttentionSpec | None = None,
) -> MiniCPMSALASparseAttentionMetadataBuilder:
    return MiniCPMSALASparseAttentionMetadataBuilder(
        kv_cache_spec=spec or _make_spec(),
        layer_names=["model.layers.0.self_attn"],
        vllm_config=None,
        device=torch.device("cpu"),
    )


class TestBuild:
    def test_maps_block_table_tensor_to_block_table(self) -> None:
        """The actual bug this builder was written to fix: real
        CommonAttentionMetadata field is `block_table_tensor`, not
        `block_table` -- every forward()/_forward_sparse() call site in
        this project references `.block_table`, so the mapping must
        happen here."""
        builder = _make_builder()
        real_block_table = torch.tensor([[3, 4, 5]], dtype=torch.int32)
        common = _make_common_attn_metadata(block_table=real_block_table)

        result = builder.build(common_prefix_len=0, common_attn_metadata=common)

        assert isinstance(result, MiniCPMSALASparseAttentionMetadata)
        assert torch.equal(result.block_table, real_block_table)

    def test_carries_dense_len_from_kv_cache_spec(self) -> None:
        spec = _make_spec()
        assert spec.dense_len == 8192
        builder = _make_builder(spec)
        common = _make_common_attn_metadata()

        result = builder.build(common_prefix_len=0, common_attn_metadata=common)

        assert result.dense_len == 8192

    def test_carries_page_block_size_from_kv_cache_spec(self) -> None:
        spec = _make_spec()
        assert spec.block_size == 256
        builder = _make_builder(spec)
        common = _make_common_attn_metadata()

        result = builder.build(common_prefix_len=0, common_attn_metadata=common)

        assert result.page_block_size == 256

    def test_preserves_seq_lens_and_query_start_loc(self) -> None:
        builder = _make_builder()
        common = _make_common_attn_metadata(seq_len=16)

        result = builder.build(common_prefix_len=0, common_attn_metadata=common)

        assert torch.equal(result.seq_lens, common.seq_lens)
        assert torch.equal(result.query_start_loc, common.query_start_loc)


class TestUseCascadeAttention:
    def test_always_returns_false(self) -> None:
        """Consistent with Stage 3b's manager (cascade/prefix-sharing
        disabled end to end for this cache type)."""
        builder = _make_builder()
        assert builder.use_cascade_attention() is False


class TestGetCudagraphSupport:
    def test_returns_never(self) -> None:
        """Honest answer: the sparse regime's data-dependent gather loop
        is fundamentally incompatible with CUDA graph capture's static-
        shape requirement."""
        spec = _make_spec()
        result = MiniCPMSALASparseAttentionMetadataBuilder.get_cudagraph_support(
            None, spec
        )
        assert result == AttentionCGSupport.NEVER


class TestUpdateBlockTable:
    def test_replaces_block_table_without_mutating_original(self) -> None:
        builder = _make_builder()
        common = _make_common_attn_metadata()
        original = builder.build(common_prefix_len=0, common_attn_metadata=common)
        original_block_table = original.block_table.clone()

        new_block_table = torch.tensor([[9, 10, 11]], dtype=torch.int32)
        new_slot_mapping = torch.arange(8, dtype=torch.int32) + 100
        updated = builder.update_block_table(
            original, new_block_table, new_slot_mapping
        )

        assert torch.equal(updated.block_table, new_block_table)
        # Original object's field must be unchanged (copy, not mutate).
        assert torch.equal(original.block_table, original_block_table)
