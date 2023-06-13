'''
This module contains the feature classes calculated from DecompData objects.
'''

import importlib

from .features import FeatureType, Features, Raws
from .means import Means
from .covariances import Covariances
from .correlations import Correlations
from .autocovariances import AutoCovariances
from .autocorrelations import AutoCorrelations
from .moup import Moup
from .cofluctuations import Cofluctuation

def from_string(feat):
    '''
    Returns the feature class given by the name `feat`, including FEATURE_ALIASES
    '''
    module = importlib.import_module("ci_lib.features")
    try:
        return getattr(module, feat)
    except AttributeError:
        pass
    return getattr(module, FEATURE_ALIASES[feat])

FEATURE_ALIASES = {
        "full-activity-dFC" : "Cofluctuation",
        "mean-activity-FC" : "Cofluctuation",
        "spot-activity-dFC" : "Cofluctuation",
        "mean" : "Means",
        "mean-activity" : "Means",
        "spot-activity" : "Means",
        "full-activity" : "Means",
        "raw" : "Raws",
        "covariance" : "Covariances",
        "correlation" : "Correlations",
        "autocovariance" : "AutoCovariances",
        "autocorrelation" : "AutoCorrelations",
        "moup" : "Moup",
        "cofluctuation" : "Cofluctuation",
        "dFC" : "Cofluctuation",
        "FC" : "Cofluctuation",
        "full-dFC" : "Cofluctuation"
}
