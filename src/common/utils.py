"""
Common utility functions used across the project.
"""
import torch
import numpy as np
from typing import Union, List, Tuple

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device(x: Union[torch.Tensor, List[torch.Tensor]], device: str) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Move tensor(s) to specified device."""
    if isinstance(x, list):
        return [to_device(xi, device) for xi in x]
    return x.to(device)

def save_checkpoint(state: dict, filename: str) -> None:
    """Save model checkpoint."""
    torch.save(state, filename)

def load_checkpoint(filename: str) -> dict:
    """Load model checkpoint."""
    return torch.load(filename) 