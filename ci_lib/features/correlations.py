import logging
import numpy as np

from .features import Features, Feature_Type
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs

LOGGER = logging.getLogger(__name__)


def calc_corrs(covs):
    var = np.diagonal(covs, axis1=1, axis2=2)
    sig = np.sqrt(var)
    return (covs / sig[:,:,None]) / sig[:,None,:]

class Correlations(Features):
    _type = Feature_Type.UNDIRECTED

    @staticmethod
    def create(data, means=None, covs=None, max_comps=None, logger=LOGGER):
        if max_comps is not None:
            logger.warn("DEPRECATED: max_comps parameter in features can not guaranty \
sensible choice of components, use n_components parameter for parcellations instead")
        if covs is None:
            if means is None:
                means = calc_means(data.temporals[:, :, :max_comps])
            elif isinstance(means, Means):
                means = means.feature
            covs = calc_covs(data.temporals[:, :, :max_comps], means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs.feature)
        feature = calc_corrs(covs)
        feat = Correlations(data.frame, data, feature)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return flat_covs(feat, diagonal=False)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
