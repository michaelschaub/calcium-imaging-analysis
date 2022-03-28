import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs


def calc_acovs(temps, means, covs, n_tau_range, label):
    n_taus = np.array(n_tau_range)
    temps = temps - means[:, None, :]
    trials, n_frames, comps = temps.shape
    cov_m = np.empty([trials, len(n_tau_range), comps, comps])

    cov_m[:, np.where(n_taus==0)[0],:,:] = covs[:,None,:,:]


    for i,i_tau in np.array(list(enumerate(n_taus)))[n_taus!=0]: #range(1, n_tau + 1):
        for trial in range(trials):
            cov_m[trial, i, :, :] = np.tensordot(temps[trial, 0:n_frames - i_tau],
                                                     temps[trial, i_tau:n_frames],
                                                     axes=(0, 0)) / float(n_frames - i_tau)
    return cov_m


DEFAULT_TIMELAG = 10


class AutoCovariances(Features):
    _type=Feature_Type.DIRECTED

    def __init__(self, data, feature, file=None, include_diagonal=True):
        super().__init__(data=data, feature=feature, file=file)
        self._include_diagonal = include_diagonal

    def create(data, means=None, covs=None, max_comps=None, max_time_lag=None, timelags=None, label = None, include_diagonal=True, logger=LOGGER):

        if max_time_lag is None or max_time_lag >= data.temporals.shape[1]:
            max_time_lag = DEFAULT_TIMELAG

        if timelags is None or np.amax(timelags) >= data.temporals.shape[1]:
            timelags = range(0, max_time_lag + 1)

        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means._feature
        if covs is None:
            covs = calc_covs(data.temporals[:, :, :max_comps], means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs._feature)

        feature = calc_acovs(data.temporals[:, :, :max_comps], means, covs, timelags, label)
        feat = AutoCovariances(data, feature, include_diagonal= include_diagonal)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate((flat_covs(feat[:, 0],diagonal=self._include_diagonal), feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
