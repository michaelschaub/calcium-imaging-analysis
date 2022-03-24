import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs


def calc_acovs(temps, means, covs, n_tau_range, label):
    temps = temps - means[:, None, :]
    trials, n_frames, comps = temps.shape
    cov_m = np.zeros([trials, len(n_tau_range)+1, comps, comps])

    cov_m[:, 0] = covs

    for trial in range(trials):

        for i,i_tau in enumerate(n_tau_range,1): #range(1, n_tau + 1):
            cov_m[trial, i, :, :] = np.tensordot(temps[trial, 0:n_frames - i_tau],
                                                     temps[trial, i_tau:n_frames],
                                                     axes=(0, 0)) / float(n_frames - i_tau)
    return cov_m


DEFAULT_TIMELAG = 10


class AutoCovariances(Features):
    _type=Feature_Type.DIRECTED

    def __init__(self, data, feature, file=None, include_diagonal=True):
        self.data = data
        self._feature = feature
        self._savefile = file
        self._include_diagonal = include_diagonal

    def create(data, means=None, covs=None, max_comps=None, max_time_lag=None, time_lag_range=None, label = None, include_diagonal=True, logger=LOGGER):
        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means._feature
        if covs is None:
            covs = calc_covs(data.temporals[:, :, :max_comps], means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs._feature)

        if max_time_lag is None or max_time_lag >= data.temporals.shape[1]:
            max_time_lag = DEFAULT_TIMELAG

        if time_lag_range is None or np.amax(time_lag_range) >= data.temporals.shape[1]:
            time_lag_range = range(1, max_time_lag + 1)

        feature = calc_acovs(data.temporals[:, :, :max_comps], means, covs, time_lag_range, label)
        feat = AutoCovariances(data, feature, include_diagonal= include_diagonal)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate((flat_covs(feat[:, 0],diagonal=self._include_diagonal), feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
