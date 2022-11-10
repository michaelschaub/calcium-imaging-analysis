import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from ci_lib import Data, DecompData


def calc_means(temps):
    return np.mean(temps, axis=1)  # average over frames


class Means(Features):
    _type = Feature_Type.NODE

    def __init__(self, data, feature, file=None, time_resolved=False):
        super().__init__(data=data, feature=feature, file=file)
        self._time_resolved = True #only needed cause it's not properly saved


    def create(data, max_comps=None, logger=LOGGER, window=1): #TODO pass window correctly from feature workflow script
        if window is None:
            feat = Means(data, feature=calc_means(data.temporals[:, :, :max_comps])) #TODO add newaxis of size 1 between trial and feat (axis=1)
        else:
            trials , phase_length, comps  =   data.temporals.shape
            windows = [range(i,i+window) for i in range(0,phase_length-window)]

            feat_val = np.zeros((trials,len(windows),comps if max_comps is None else max_comps))
            for w,window in enumerate(windows):
                feat_val[:,w,:] = calc_means(data.temporals[:, window, :max_comps])
                
            feat = Means(data, feature=feat_val, time_resolved=True)
            print(feat.feature.shape)

        return feat

    def flatten(self, timepoints=slice(None), feat=None):
        if feat is None:
            feat = self._feature
        print(feat.shape)

        feat = feat[:,timepoints,:]
        mask = np.ones((feat.shape[1:]), dtype=bool)
        return feat[:,mask]


    @property
    def pixel(self):
        return DecompData.PixelSlice(self._feature, self.data._spats[:self._feature.shape[1]])

    def _op_data(self, a):
        df = self.data._df
        if isinstance(a, Data):
            temps = self.expand(a)
        else:
            temps = self.expand()
        spats = self.data.spatials
        starts = self.data._starts
        return df, temps, spats, starts

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
