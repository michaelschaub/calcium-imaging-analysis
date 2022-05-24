import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from .means import Means, calc_means


def calc_covs(temps, means):
    # TODO: Optimize, currently calculates off diagonals double
    temps = temps - means[:, None, :]
    return np.einsum("itn,itm->inm", temps, temps) / temps.shape[1]


def flat_covs(covs, diagonal):
    # true upper triangle matrix (same shape as covariance)
    ind = np.triu(np.ones(covs.shape[1:], dtype=bool),k=int(not diagonal))
    # flattened upper triangle of covariances
    return covs[:, ind]


class Covariances(Features):
    _type = Feature_Type.UNDIRECTED

    def __init__(self, data, feature, file=None, include_diagonal=True):
        super().__init__(data=data, feature=feature, file=file)
        self._include_diagonal = include_diagonal

    def create(data, means=None, max_comps=None, include_diagonal=True, logger=LOGGER):
        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means._feature
        feature = calc_covs(data.temporals[:, :, :max_comps], means)
        feat = Covariances(data, feature, include_diagonal=include_diagonal)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return flat_covs(feat, self._include_diagonal)

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
