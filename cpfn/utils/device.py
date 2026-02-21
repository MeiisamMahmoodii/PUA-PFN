"""
Device management utilities
"""

import torch


def get_device(device: str = "auto") -> str:
    """
    Get device to use for training.

    Args:
        device: "cuda", "cpu", or "auto"

    Returns:
        device string
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
