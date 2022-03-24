import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs
from .autocovariances import AutoCovariances, calc_acovs, DEFAULT_TIMELAG


def calc_acorrs(acovs):
    var = np.diagonal(acovs[:,0,:,:], axis1=1, axis2=2)
    sig = np.sqrt(var)
    return (acovs / sig[:,None,:,None]) / sig[:,None,None,:]

class AutoCorrelations(Features):
    _type = Feature_Type.UNDIRECTED

    def __init__(self, data, feature, file=None):
        self.data = data
        self._feature = feature
        self._savefile = file

    def create(data, means=None, covs=None, acovs=None, max_comps=None, max_time_lag=None, time_lag_range=None, label = None, logger=LOGGER):
        if max_time_lag is None or max_time_lag >= data.temporals.shape[1]:
            max_time_lag = DEFAULT_TIMELAG

        if time_lag_range is None or np.amax(time_lag_range) >= data.temporals.shape[1]:
            time_lag_range = range(1, max_time_lag + 1)

        if acovs is None:
            if covs is None:
                if means is None:
                    means = calc_means(data.temporals[:, :, :max_comps])
                elif isinstance(means, Means):
                    means = means._feature
                covs = calc_covs(data.temporals[:, :, :max_comps], means)
            elif isinstance(covs, Covariances):
                covs = np.copy(covs._feature)
            acovs = calc_acovs(data.temporals[:, :, :max_comps], means, covs, time_lag_range, label)
        elif isinstance(acovs, AutoCovariances):
            acovs = np.copy(acovs._feature)
        feature = calc_acorrs(acovs)
        feat = AutoCorrelations(data, feature)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate((flat_covs(feat[:, 0],diagonal=False), feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
