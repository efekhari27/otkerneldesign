"""otkerneldesign module."""

from .KernelHerding import KernelHerding
from .KernelHerdingTensorized import KernelHerdingTensorized
from .TestSetWeighting import TestSetWeighting

__all__ = [
    "KernelHerding",
    "TestSetWeighting",
    "KernelHerdingTensorized",
]
__version__ = "0.1.2"
