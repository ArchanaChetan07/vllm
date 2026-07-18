# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for sparse_config parsing (H3)."""

import pytest
from transformers import PretrainedConfig

from vllm.v1.attention.backends.minicpm_sala_sparse import (
    MiniCPMSALASparseConfig,
    parse_sparse_config,
    validate_page_block_size,
)

RELEASED_SPARSE_CONFIG = {
    "kernel_size": 32,
    "kernel_stride": 16,
    "init_blocks": 1,
    "block_size": 64,
    "window_size": 2048,
    "topk": 64,
    "dense_len": 8192,
}


def test_parse_sparse_config_from_hf_config() -> None:
    cfg = PretrainedConfig(sparse_config=RELEASED_SPARSE_CONFIG)
    sc = parse_sparse_config(cfg)
    assert sc.kernel_size == 32
    assert sc.kernel_stride == 16
    assert sc.dense_len == 8192
    assert sc.topk == 64
    assert sc.sparse_block_size == 64
    assert sc.local_blocks == 32
    # Reference: MiniCPMInfLLMv2Attention.__init__ budgets the local-window
    # blocks ON TOP of the configured topk (64 + 2048//64 = 96). Passing the
    # raw config topk to compressed_attention would select 32 fewer blocks.
    assert sc.effective_topk == 96


def test_tier2_derivation_is_4x() -> None:
    sc = MiniCPMSALASparseConfig(
        kernel_size=32,
        kernel_stride=16,
        dense_len=8192,
        init_blocks=1,
        topk=64,
        window_size=2048,
        sparse_block_size=64,
    )
    assert sc.compress_k2_kernel_size == 128
    assert sc.compress_k2_kernel_stride == 64


def test_missing_sparse_config_raises() -> None:
    cfg = PretrainedConfig()
    with pytest.raises(ValueError, match="sparse_config"):
        parse_sparse_config(cfg)


def test_invalid_window_size_raises() -> None:
    bad = {**RELEASED_SPARSE_CONFIG, "window_size": 2000}
    cfg = PretrainedConfig(sparse_config=bad)
    with pytest.raises(ValueError, match="divisible"):
        parse_sparse_config(cfg)


def test_validate_page_block_size() -> None:
    validate_page_block_size(256)
    with pytest.raises(ValueError, match="multiple of 256"):
        validate_page_block_size(128)
    with pytest.raises(ValueError, match="positive"):
        validate_page_block_size(0)
