'''
This module contains the temporal means feature.
'''

import logging
import numpy as np

from ci_lib import DecompData
from .features import Features, FeatureType

LOGGER = logging.getLogger(__name__)


def calc_means(temps):
    '''Calculate the temporal means from temporal components'''
    return np.mean(temps, axis=1) # average over frames


class Means(Features):
    '''
    A feature containing the mean of the temporal components over whole trials
    or rolling window with the trials
    '''

    _type = FeatureType.NODE

    def __init__(self, frame, data, feature, file=None, time_resolved=False, full=False):
        super().__init__(frame=frame, data=data, feature=feature, file=file, full=full)
        self._time_resolved = time_resolved #only needed cause it's not properly saved


    @staticmethod
    def create(data, logger=LOGGER, window=None, start=None, stop=None,
               full=False,z_scored=True): #TODO z_score default ot False
        '''Create this feature from a DecompData object'''

        if window is None:
            #TODO start:stop should be supported by window as well
            feat = Means(data.frame, data, feature=calc_means(
                            data.temporals[:, slice(start,stop), :])[:,np.newaxis,:])
        else:
            trials, phase_length, comps = data.temporals[:, slice(start,stop), :].shape
            windows = [range(i,i+window) for i in range(0,phase_length-window+1)]

            feat_val = np.zeros((trials,len(windows),comps ))
            for w_indx,window in enumerate(windows):
                if not z_scored:
                    temps = data.temporals[:, slice(start,stop), :][:, window, :]
                else:
                    temps = data.temporals_z_scored[:, slice(start,stop), :][:, window, :]
                feat_val[:,w_indx,:] = calc_means(temps)
            feat = Means(data.frame, data, feature=feat_val, time_resolved=True,full=full)

        return feat

    def flatten(self, timepoints=slice(None), feat=None):
        '''
        Flattens the feature into one trial dimension and one dimension for everything else
        '''
        if feat is None:
            feat = self._feature

        feat = feat[:,timepoints,:]
        mask = np.ones((feat.shape[1:]), dtype=bool)
        return feat[:,mask]


    @property
    def pixel(self):
        '''
        Creates PixelSlice object, from which slices of recomposed pixel data can be accessed
        the first key is applied to the trials, the second and third to the horizontal
        and vertical dimension of the spatials
        '''
        return DecompData.PixelSlice(self._feature, self.data.spatials[:self._feature.shape[1]])

    @property
    def ncomponents(self):
        '''The number of components the data is decomposed into'''
        return self._feature.shape[-1]
