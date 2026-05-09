"""
Compatibility imports for deep learning models.

The concrete implementations live in dedicated modules:
- utils.models.cnn
- utils.models.recurrent
"""

from .cnn import ConvNet1D, ConvNetModel
from .deep_common import get_torch_device, plot_learning_curve
from .recurrent import RecurrentNet, RecurrentNetModel

__all__ = [
    "ConvNet1D",
    "ConvNetModel",
    "RecurrentNet",
    "RecurrentNetModel",
    "get_torch_device",
    "plot_learning_curve",
]
