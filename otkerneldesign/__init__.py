"""otkerneldesign module."""

from .GreedySupportPoints import GreedySupportPoints
from .KernelHerding import KernelHerding
from .KernelHerdingTensorized import KernelHerdingTensorized
from .TestSetWeighting import TestSetWeighting
from .BayesianQuadraturetWeighting import BayesianQuadraturetWeighting

__all__ = [
    "GreedySupportPoints",
    "KernelHerding",
    "KernelHerdingTensorized",
    "TestSetWeighting",
    "BayesianQuadraturetWeighting",
]
__version__ = "0.1.2"
