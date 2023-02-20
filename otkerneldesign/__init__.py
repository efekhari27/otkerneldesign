"""otkerneldesign module."""

from .GreedySupportPoints import GreedySupportPoints
from .KernelHerding import KernelHerding
from .KernelHerdingTensorized import KernelHerdingTensorized
from .TestSetWeighting import TestSetWeighting
from .BayesianQuadrature import BayesianQuadrature

__all__ = [
    "GreedySupportPoints",
    "KernelHerding",
    "KernelHerdingTensorized",
    "TestSetWeighting",
    "BayesianQuadrature",
]
__version__ = "0.1.5"
