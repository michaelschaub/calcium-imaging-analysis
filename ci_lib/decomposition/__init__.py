'''
This module contains functions for calculating variously decomposed DecompData
from SVD decomposed DecompData objects
'''

from .decomposition import anatomical_parcellation, fastICA
from .locanmf import locaNMF
from .svd import svd, blockwise_svd, postprocess_SVD
