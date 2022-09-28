from ._base import IntervalSelector, PointSelector, SpectralSelector, Convertible, FeatureDescriptor
from ._bipls import BiPLS
from ._cars import CARS
from ._fipls import FiPLS
from ._ipls import IPLS
from ._mcuve import MCUVE
from ._pseudo_interval import PseudoIntervalSelector
from ._random_frog import IntervalRandomFrog
from ._random_frog import RandomFrog
from ._spa import SPA
from ._vip import VIP
from ._vip_spa import VIP_SPA
from ._vissa import VISSA
from .util import optimize_intervals

__version__ = '0.9.0'

__all__ = [
    'PointSelector',
    'IntervalSelector',
    'SpectralSelector',
    'Convertible',
    'FeatureDescriptor',
    'PseudoIntervalSelector',
    'CARS',
    'MCUVE',
    'RandomFrog',
    'SPA',
    'VIP',
    'VIP_SPA',
    'VISSA',
    'IntervalRandomFrog',
    'IPLS',
    'FiPLS',
    'BiPLS',
    'optimize_intervals'
]
