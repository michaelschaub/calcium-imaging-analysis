from .features import Feature_Type, Features, Raws
from .means import Means
from .covariances import Covariances
from .correlations import Correlations
from .autocovariances import AutoCovariances
from .autocorrelations import AutoCorrelations
from .moup import Moup

import importlib

def from_string(feat):
    module = importlib.import_module("ci_lib.features")
    return getattr(module, feat)
