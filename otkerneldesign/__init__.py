"""otkerneldesign module."""

from .GreedySupportPoints import GreedySupportPoints
from .KernelHerding import KernelHerding
from .KernelHerdingTensorized import KernelHerdingTensorized
from .TestSetWeighting import TestSetWeighting

__all__ = [
    "GreedySupportPoints",
    "KernelHerding",
    "KernelHerdingTensorized",
    "TestSetWeighting",
]
__version__ = "0.1.2"
