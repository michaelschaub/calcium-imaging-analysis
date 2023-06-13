import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means
from .covariances import Covariances, flat_covs
from .autocovariances import AutoCovariances


def calc_acorrs(covs, acovs):
    var = np.diagonal(covs, axis1=1, axis2=2)
    sig = np.sqrt(var)
    return (acovs / sig[:,None,:,None]) / sig[:,None,None,:]

class AutoCorrelations(Features):
    _type = Feature_Type.UNDIRECTED

    @staticmethod
    def create(data, means=None, covs=None, acovs=None, max_comps=None, timelag=1, logger=LOGGER):
        if max_comps is not None:
            logger.warn("DEPRECATED: max_comps parameter in features can not garanty sensible choice of components, use n_components parameter for parcellations instead")
        if covs is None:
            covs = Covariances.create(data, means, max_comps, True, logger).feature
        elif isinstance(covs, Covariances):
            covs = np.copy(acovs.feature)
        if acovs is None:
            acovs = AutoCovariances.create(data, means, covs, max_comps, timelag, True, logger).feature
        elif isinstance(acovs, AutoCovariances):
            acovs = np.copy(acovs.feature)

        feature = calc_acorrs(covs, acovs)
        feat = AutoCorrelations(data.frame, data, feature)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate((flat_covs(feat[:, 0],diagonal=False), feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
