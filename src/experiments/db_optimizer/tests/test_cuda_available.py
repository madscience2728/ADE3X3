"""Test CUDA availability and basic GPU operations."""

import pytest
import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA not available"


def test_gpu_device_name():
    name = torch.cuda.get_device_name(0)
    assert "3060" in name or len(name) > 0, f"Unexpected GPU: {name}"


def test_gpu_fp32_basic():
    a = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    b = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    c = a @ b
    assert c.shape == (100, 100)
    assert c.dtype == torch.float32


def test_gpu_fp64_basic():
    a = torch.randn(10, 10, device="cuda", dtype=torch.float64)
    b = torch.randn(10, 10, device="cuda", dtype=torch.float64)
    c = a @ b
    assert c.shape == (10, 10)
    assert c.dtype == torch.float64


def test_gpu_vram_sufficient():
    total = torch.cuda.get_device_properties(0).total_memory
    assert total >= 10 * (1024 ** 3), f"Need >=10 GB VRAM, got {total / (1024**3):.1f} GB"
