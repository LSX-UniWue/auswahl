from ._base import IntervalSelector, PointSelector, SpectralSelector, FeatureDescriptor, Selection
from ._cars import CARS
from ._ipls import IPLS
from ._mcuve import MCUVE
from ._random_frog import IntervalRandomFrog
from ._random_frog import RandomFrog
from ._spa import SPA
from ._version import __version__
from ._vip import VIP
from ._vip_spa import VIP_SPA
from ._vissa import VISSA, iVISSA
from ._fipls import FiPLS
from ._bipls import BiPLS
from ._dummy import DummyIntervalSelector, DummyPointSelector, ExceptionalSelector

__all__ = ['PointSelector', 'IntervalSelector',
           'CARS', 'MCUVE', 'RandomFrog', 'SPA', 'VIP', 'VIP_SPA', 'VISSA',
           'IntervalRandomFrog', 'IPLS', 'FiPLS', 'BiPLS', 'iVISSA',
           '__version__']
