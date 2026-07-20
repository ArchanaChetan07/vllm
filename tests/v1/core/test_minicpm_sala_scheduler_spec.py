# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler integration: sparse Attention returns the
HierarchicalCompressedAttentionSpec (C1)."""

from unittest.mock import MagicMock

import torch

from vllm.model_executor.models.minicpm_sala_sparse_wiring import (
    MiniCPMSALASparseAttention,
)
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.minicpm_sala_sparse import MiniCPMSALASparseConfig
from vllm.v1.core.minicpm_sala_kv_cache_spec import (
    HierarchicalCompressedAttentionManager,
    HierarchicalCompressedAttentionSpec,
    build_hierarchical_compressed_attention_spec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

RELEASED_SPARSE = MiniCPMSALASparseConfig(
    kernel_size=32,
    kernel_stride=16,
    dense_len=8192,
    init_blocks=1,
    topk=64,
    window_size=2048,
    sparse_block_size=64,
)


class TestBuildHierarchicalCompressedAttentionSpec:
    def test_fields_match_sparse_config(self) -> None:
        spec = build_hierarchical_compressed_attention_spec(
            block_size=256,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            compress_kernel_size=RELEASED_SPARSE.kernel_size,
            compress_kernel_stride=RELEASED_SPARSE.kernel_stride,
            dense_len=RELEASED_SPARSE.dense_len,
        )
        assert isinstance(spec, HierarchicalCompressedAttentionSpec)
        assert spec.block_size == 256
        assert spec.dense_len == 8192
        assert spec.compress_kernel_size == 32


class TestRegistryAcceptsSparseSpec:
    def test_registered_manager_is_hierarchical(self) -> None:
        spec = build_hierarchical_compressed_attention_spec(
            block_size=256,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            compress_kernel_size=32,
            compress_kernel_stride=16,
            dense_len=8192,
        )
        KVCacheSpecRegistry.check_kv_cache_spec_registry(
            {"model.layers.0.self_attn.attn": spec}
        )
        manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
        assert manager_cls is HierarchicalCompressedAttentionManager


class TestMiniCPMSALASparseAttentionGetKvCacheSpec:
    def test_returns_hierarchical_spec_not_full_attention(self) -> None:
        attn = MiniCPMSALASparseAttention.__new__(MiniCPMSALASparseAttention)
        attn.attn_type = AttentionType.DECODER
        attn.num_kv_heads = 2
        attn.head_size = 128
        attn.kv_cache_dtype = "auto"
        attn.kv_cache_torch_dtype = torch.bfloat16
        attn._sparse_config = RELEASED_SPARSE

        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 256

        spec = attn.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, HierarchicalCompressedAttentionSpec)
        assert spec.dense_len == 8192
        assert spec.block_size == 256
