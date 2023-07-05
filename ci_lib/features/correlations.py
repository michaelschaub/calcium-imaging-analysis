'''
This module contains the component correlations feature.
'''

import logging
import numpy as np

from .features import Features, FeatureType
from .means import Means, calc_means
from .covariances import Covariances, calc_covs, flat_covs

LOGGER = logging.getLogger(__name__)


def calc_corrs(covs):
    '''Calculate the component correlations from covariances'''
    var = np.diagonal(covs, axis1=1, axis2=2)
    sig = np.sqrt(var)
    return (covs / sig[:,:,None]) / sig[:,None,:]

class Correlations(Features):
    '''
    A feature containing the component correlations within single trials
    '''

    _type = FeatureType.UNDIRECTED

    @staticmethod
    def create(data, means=None, covs=None, logger=LOGGER):
        '''Create this feature from a DecompData object'''

sensible choice of components, use n_components parameter for parcellations instead")
        if covs is None:
            if means is None:
                means = calc_means(data.temporals)
            elif isinstance(means, Means):
                means = means.feature
            covs = calc_covs(data.temporals, means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs.feature)
        feature = calc_corrs(covs)
        feat = Correlations(data.frame, data, feature)
        return feat

    def flatten(self, feat=None):
        '''
        Flattens the feature into one trial dimension and one dimension for everything else
        '''
        if feat is None:
            feat = self._feature
        return flat_covs(feat, diagonal=False)

    @property
    def ncomponents(self):
        '''The number of components the data is decomposed into'''
        return self._feature.shape[-1]
