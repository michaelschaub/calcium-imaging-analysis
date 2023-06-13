'''
This module contains the component covariances feature.
'''

import logging
import numpy as np

from .features import Features, Feature_Type
from .means import Means, calc_means

LOGGER = logging.getLogger(__name__)


def calc_covs(temps, means):
    '''Calculate the component covariances from temporal components and means'''
    # TODO: Optimize, currently calculates off diagonals double
    temps = temps - means[:, None, :]
    return np.einsum("itn,itm->inm", temps, temps) / temps.shape[1]


def flat_covs(covs, diagonal):
    '''
    Flatten the covariances without redundance
    '''
    # true upper triangle matrix (same shape as covariance)
    ind = np.triu(np.ones(covs.shape[1:], dtype=bool),k=int(not diagonal))
    # flattened upper triangle of covariances
    return covs[:, ind]


class Covariances(Features):
    '''
    A feature containing the component covariances within single trials
    '''

    _type = Feature_Type.UNDIRECTED

    def __init__(self, frame, data, feature, file=None, include_diagonal=True):
        super().__init__(frame=frame, data=data, feature=feature, file=file)
        self._include_diagonal = include_diagonal

    @staticmethod
    def create(data, means=None, max_comps=None, include_diagonal=True, logger=LOGGER):
        '''Create this feature from a DecompData object'''
        if max_comps is not None:
            logger.warn("DEPRECATED: max_comps parameter in features can not guaranty \
sensible choice of components, use n_components parameter for parcellations instead")
        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means.feature
        feature = calc_covs(data.temporals[:, :, :max_comps], means)
        feat = Covariances(data.frame, data, feature, include_diagonal=include_diagonal)
        return feat

    def flatten(self, feat=None):
        '''
        Flattens the feature into one trial dimension and one dimension for everything else
        '''
        if feat is None:
            feat = self._feature
        return flat_covs(feat, self._include_diagonal)

    @property
    def ncomponents(self):
        '''The number of components the data is decomposed into'''
        return self._feature.shape[-1]
