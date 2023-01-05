import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from ci_lib import Data, DecompData


def calc_means(temps):
    return np.mean(temps, axis=1) # average over frames


class Means(Features):
    _type = Feature_Type.NODE

    def __init__(self, data, feature, file=None, time_resolved=False, full=False):
        super().__init__(data=data, feature=feature, file=file, full=full)
        self._time_resolved = True #only needed cause it's not properly saved


    def create(data, max_comps=None, logger=LOGGER, window=None, start=None, stop=None,full=False): 
        if window is None:
            feat = Means(data, feature=calc_means(data.temporals[:, slice(start,stop), :max_comps])[:,np.newaxis,:])  #TODO start:stop should be supported by window as well
        else:
            trials , phase_length, comps  =   data.temporals[:, slice(start,stop), :max_comps].shape
            windows = [range(i,i+window) for i in range(0,phase_length-window)]

            feat_val = np.zeros((trials,len(windows),comps if max_comps is None else max_comps))
            for w,window in enumerate(windows):
                feat_val[:,w,:] = calc_means(data.temporals[:, slice(start,stop), :][:, window, :max_comps])
                
            feat = Means(data, feature=feat_val, time_resolved=True,full=full)

        return feat

    def flatten(self, timepoints=slice(None), feat=None):
        if feat is None:
            feat = self._feature

        feat = feat[:,timepoints,:]
        mask = np.ones((feat.shape[1:]), dtype=bool)
        return feat[:,mask]

    def mean(self):
        return Means(self._data, feature=calc_means(self._feature[:, :, :]))


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
