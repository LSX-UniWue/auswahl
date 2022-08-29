
from ._base import IntervalSelector, PointSelector, SpectralSelector, Convertible, FeatureDescriptor
from .benchmarking._util import Selection
from ._cars import CARS
from ._ipls import IPLS
from ._mcuve import MCUVE
from ._random_frog import IntervalRandomFrog
from ._random_frog import RandomFrog
from ._spa import SPA
from ._vip import VIP
from ._vip_spa import VIP_SPA
from ._vissa import VISSA
from ._fipls import FiPLS
from ._bipls import BiPLS
from ._pseudo_interval import PseudoIntervalSelector
from ._dummy import DummyIntervalSelector, DummyPointSelector, ExceptionalSelector
from .util import optimize_intervals

__version__ = '0.0.1'

__all__ = ['PointSelector', 'IntervalSelector','SpectralSelector', 'Convertible', 'FeatureDescriptor',
           'CARS', 'MCUVE', 'RandomFrog', 'SPA', 'VIP', 'VIP_SPA', 'VISSA',
           'IntervalRandomFrog', 'IPLS', 'FiPLS', 'BiPLS',
           '__version__', 'optimize_intervals']
