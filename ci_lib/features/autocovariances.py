'''
This module contains the (time-lagged) auto-covariances feature.
'''

import logging
import numpy as np

from ci_lib.plotting import plot_connectivity_matrix

from .features import Features, FeatureType
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs

LOGGER = logging.getLogger(__name__)


def calc_acovs(temps, means, covs, n_tau_range):
    '''
    Calculate the auto-covariances from temporal components, means
    and covariances (avoid duplicate calculation)
    '''
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


class AutoCovariances(Features):
    '''
    A feature containing the time-lagged auto-covariances within single trials
    '''

    _type=FeatureType.DIRECTED

    def __init__(self, frame, data, feature, file=None, include_diagonal=True):
        super().__init__(frame=frame, data=data, feature=feature, file=file)
        self._include_diagonal = include_diagonal

    @staticmethod
    def create(data, means=None, covs=None, timelag=1, include_diagonal=True,
               logger=LOGGER):
        '''Create this feature from a DecompData object'''

sensible choice of components, use n_components parameter for parcellations instead")

        timelags = np.asarray(timelag, dtype=int).reshape(-1)
        if np.max(timelags) >= data.temporals.shape[1]:
            logger.warning("AutoCovariances with timelag exceeding length of data found, \
removing too large timelags!")
            timelags = timelags[timelags >= data.temporals.shape[1]]
        if len(timelags) == 0:
            raise ValueError

        if means is None:
            means = calc_means(data.temporals)
        elif isinstance(means, Means):
            means = means.feature
        if covs is None:
            covs = calc_covs(data.temporals, means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs.feature)

        feature = calc_acovs(data.temporals, means, covs, timelags)
        feat = AutoCovariances(data.frame, data, feature, include_diagonal= include_diagonal)
        return feat

    def flatten(self, feat=None):
        '''
        Flattens the feature into one trial dimension and one dimension for everything else
        '''
        if feat is None:
            feat = self._feature
        return np.concatenate(
                (flat_covs(feat[:, 0],
                           diagonal=self._include_diagonal),
                 feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    @property
    def ncomponents(self):
        '''The number of components the data is decomposed into'''
        return self._feature.shape[-1]

    def plot(self,path):
        #TODO why is it trial x 1 (?) x w x h
        plot_connectivity_matrix(
                [np.mean(self._feature,axis=0)[0],np.std(self._feature,axis=0)[0]],
                title="mean|std",path=path)
