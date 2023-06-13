import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs

from ci_lib.plotting import plot_connectivity_matrix


def calc_acovs(temps, means, covs, taus, label):
    #Taus either scalar or array of lags
    n_taus = len(tau) if isinstance(tau,(list,np.ndarray)) else 1
    taus = np.array(taus) 
    temps = temps - means[:, None, :]
    trials, n_frames, comps = temps.shape
    cov_m = np.empty([trials, n_taus, comps, comps])

    cov_m[:, np.where(taus==0)[0],:,:] = covs[:,None,:,:] #TODO likely breaks for taus being scalar


    for i,tau in np.array(list(enumerate(taus)))[taus!=0]: #range(1, n_tau + 1):
        for trial in range(trials):
            cov_m[trial, i, :, :] = np.tensordot(temps[trial, 0:n_frames - tau],
                                                     temps[trial, tau:n_frames],
                                                     axes=(0, 0)) / float(n_frames - tau)
    return cov_m


class AutoCovariances(Features):
    _type=Feature_Type.DIRECTED

    def __init(self, frame, data, feature, file=None, include_diagonal=True):
        super().__init(frame=frame, data=data, feature=feature, file=file)
        self._include_diagonal = include_diagonal

    @staticmethod
    def create(data, means=None, covs=None, max_comps=None, timelag=None, label = None, include_diagonal=True, logger=LOGGER):

        if timelag is None or timelag >= data.temporals.shape[1]:
            timelag = 0

        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means._feature
        if covs is None:
            covs = calc_covs(data.temporals[:, :, :max_comps], means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs._feature)

        feature = calc_acovs(data.temporals[:, :, :max_comps], means, covs, timelag, label)

        feat = AutoCovariances(data.frame, data, feature, include_diagonal= include_diagonal)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate((flat_covs(feat[:, 0],diagonal=self._include_diagonal), feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]

    def plot(self,path):
        plot_connectivity_matrix([np.mean(self._feature,axis=0)[0],np.std(self._feature,axis=0)[0]],title="mean|std",path=path) #TODO why is it trial x 1 (?) x w x h

