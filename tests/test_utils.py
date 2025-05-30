"""
Unit tests for common utilities.
"""
import pytest
import torch
import numpy as np
from src.common.utils import set_seed, to_device, save_checkpoint, load_checkpoint

def test_set_seed():
    """Test that set_seed makes results reproducible."""
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)

def test_to_device():
    """Test tensor device movement."""
    x = torch.randn(10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_device = to_device(x, device)
    assert x_device.device.type == device

def test_checkpoint_io(tmp_path):
    """Test checkpoint saving and loading."""
    state = {"model": torch.randn(10), "optimizer": torch.randn(5)}
    filename = tmp_path / "test_checkpoint.pt"
    
    save_checkpoint(state, str(filename))
    loaded_state = load_checkpoint(str(filename))
    
    assert torch.allclose(state["model"], loaded_state["model"])
    assert torch.allclose(state["optimizer"], loaded_state["optimizer"]) 