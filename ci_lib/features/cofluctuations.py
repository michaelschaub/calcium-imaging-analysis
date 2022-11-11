import numpy as np
import scipy.stats

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means, calc_means


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

    def __init__(self, data, feature, file=None, include_diagonal=True):
        super().__init__(data=data, feature=feature, file=file)
        self._time_resolved = True
        self._include_diagonal = include_diagonal

    def create(data, max_comps=None, include_diagonal=True, logger=LOGGER):
        #zscores_over_time = scipy.stats.zscore(data.temporals[:, :, :max_comps],axis=1) 
        zscores_over_time = data.temporals_z_scored[:, :, :max_comps]

        trials, frames, comps = zscores_over_time.shape
        co_fluct = np.zeros((trials, frames, comps, comps))
        co_fluct = np.einsum('...n,...m->...nm',zscores_over_time,zscores_over_time)

        # [np.matmul(f,np.transpose(f)) for f in t] for t in zscores_over_time]
            
        #co_fluct_slow = np.array([[i*j for j in b.T]  for i in a.T])
        #co_fluct = zscores_over_time[:,:,:,None] * zscores_over_time[:,:,None,:]

        #np.einsum('ijk,ijl->ijkl',zscores_over_time,zscores_over_time)
        #np.tensordot(zscores_over_time,np.transpose(zscores_over_time),axis=)

        rss = np.sqrt(np.sum(co_fluct*co_fluct,axis=(2,3)))


        return Cofluctuation(data, co_fluct,include_diagonal=include_diagonal) #, )

    def flatten(self, timepoints=slice(None), feat=None):
        if feat is None:
            feat = self._feature
        return flat_time_resolved_c(feat, self._include_diagonal,timepoints)



    @property
    def ncomponents(self):
        return self._feature.shape[-1]
