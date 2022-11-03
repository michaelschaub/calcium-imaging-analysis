import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from ci_lib import Data, DecompData


def calc_means(temps):
    return np.mean(temps, axis=1)  # average over frames


class Means(Features):
    _type = Feature_Type.NODE

    def create(data, max_comps=None, logger=LOGGER, window_size=None):
        if window is None:
            feat = Means(data, feature=calc_means(data.temporals[:, :, :max_comps]))
        else:
            trials , phase_length, comps  =   data.temporals.shape
            windows = [range(i,i+window_size) for i in range(0,phase_length-window_size)]

            feat_val = np.zeros((len(windows),trials,comps if max_comps is None else max_comps))
            for w,window in enumerate(windows):
                feat_val[w] = calc_means(data.temporals[:, window, :max_comps])
                
            feat = Means(data, feature=feat_val, time_resolved=True)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return feat

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
