"""otkerneldesign module."""

from .GreedySupportPoints import GreedySupportPoints
from .KernelHerding import KernelHerding
from .KernelHerdingTensorized import KernelHerdingTensorized
from .TestSetWeighting import TestSetWeighting
from .BayesianQuadratureWeighting import BayesianQuadratureWeighting

__all__ = [
    "GreedySupportPoints",
    "KernelHerding",
    "KernelHerdingTensorized",
    "TestSetWeighting",
    "BayesianQuadratureWeighting",
]
__version__ = "0.1.4"
