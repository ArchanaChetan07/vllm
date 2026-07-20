# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR2-only sparse attention wiring for MiniCPM-SALA."""

from __future__ import annotations

from transformers import PretrainedConfig

import vllm.v1.core.minicpm_sala_kv_cache_spec  # noqa: F401
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.minicpm_sala_sparse import (
    INFLLM_V2_AVAILABLE,
    MiniCPMSALASparseAttentionBackend,
    MiniCPMSALASparseConfig,
    parse_sparse_config,
    validate_page_block_size,
)
from vllm.v1.core.minicpm_sala_kv_cache_spec import (
    build_hierarchical_compressed_attention_spec,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, get_kv_quant_mode


class MiniCPMSALASparseAttention(Attention):
    """Sparse-layer Attention reporting HierarchicalCompressedAttentionSpec."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        *,
        sparse_config: MiniCPMSALASparseConfig,
        num_kv_heads: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self._sparse_config = sparse_config
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_backend=MiniCPMSALASparseAttentionBackend,
            # Reaches MiniCPMSALASparseAttentionImpl.__init__ via
            # Attention's **extra_impl_args passthrough -- the impl
            # requires it as a keyword-only argument.
            sparse_config=sparse_config,
            **extra_impl_args,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        assert self.attn_type == AttentionType.DECODER
        block_size = vllm_config.cache_config.block_size
        validate_page_block_size(block_size)
        sc = self._sparse_config
        return build_hierarchical_compressed_attention_spec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            compress_kernel_size=sc.kernel_size,
            compress_kernel_stride=sc.kernel_stride,
            dense_len=sc.dense_len,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )


def create_sparse_attention_if_available(
    config: PretrainedConfig,
    *,
    num_heads: int,
    head_dim: int,
    scaling: float,
    num_kv_heads: int,
    cache_config: CacheConfig | None,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> Attention | None:
    """Return sparse Attention when infllm_v2 is available; else None."""
    if not INFLLM_V2_AVAILABLE:
        return None
    if cache_config is None:
        raise ValueError(
            "cache_config is required when infllm_v2 sparse backend is active"
        )
    validate_page_block_size(cache_config.block_size)
    sparse_config = parse_sparse_config(config)
    return MiniCPMSALASparseAttention(
        num_heads,
        head_dim,
        scaling,
        num_kv_heads=num_kv_heads,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=prefix,
        sparse_config=sparse_config,
        block_size=cache_config.block_size,
    )
