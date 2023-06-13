import logging
import numpy as np
import scipy.stats

from .features import Features, Feature_Type
from .means import Means, calc_means

LOGGER = logging.getLogger(__name__)


def calc_covs(temps, means):
    # TODO: Optimize, currently calculates off diagonals double
    temps = temps - means[:, None, :]
    return np.einsum("itn,itm->inm", temps, temps) / temps.shape[1]


def flat_time_resolved_c(connectivity, diagonal,timepoints=slice(None)):
    #Select timepoints
    connectivity = connectivity[:,timepoints]


    # true upper triangle matrix (same shape as covariance)
    ind = np.triu(np.ones(connectivity.shape[1:], dtype=bool),k=int(not diagonal))
    # flattened upper triangle of all trials and timepoints

    return connectivity[:, ind] 


class Cofluctuation(Features):
    _type = Feature_Type.UNDIRECTED

    def __init__(self, frame, data, feature, file=None, include_diagonal=True, range=None, full=False):
        super().__init__(frame=frame, data=data, feature=feature, file=file,full=full)
        self._time_resolved = True
        self._include_diagonal = include_diagonal
        self._range = range if range else (0,feature.shape[1])

    @staticmethod
    def create(data, max_comps=None, include_dia=False, logger=LOGGER,window=None,mean=False,start=None,stop=None,full=False):
        zscores_over_time = data.temporals_z_scored[:,  slice(start,stop), :max_comps]

        trials, frames, comps = zscores_over_time.shape
        co_fluct = np.zeros((trials, frames, comps, comps))
        co_fluct = np.einsum('...n,...m->...nm',zscores_over_time,zscores_over_time)

        rss = np.sqrt(np.sum(co_fluct*co_fluct,axis=(2,3)))
        logger.info(f"RSS: {rss}")

        if mean:
            #Take mean over all frames from start to stop
            co_fluct = np.mean(co_fluct,axis=1)[:,np.newaxis,:]

        #if start is None:
        #    start = 0
        #if stop is None:
        #    stop = frames-1

        return Cofluctuation(data.frame, data, co_fluct,include_diagonal=include_dia, full=full) #, )

    def flatten(self, timepoints=slice(None), feat=None):
        if feat is None:
            feat = self._feature
        return flat_time_resolved_c(feat, self._include_diagonal,timepoints)



    @property
    def ncomponents(self):
        return self._feature.shape[-1]
