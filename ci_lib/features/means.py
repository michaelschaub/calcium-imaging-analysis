import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from ci_lib import Data, DecompData


def calc_means(temps):
    return np.mean(temps, axis=1)  # average over frames


class Means(Features):
    _type = Feature_Type.NODE

    def create(data, max_comps=None, logger=LOGGER):
        feat = Means(data, feature=calc_means(data.temporals[:, :, :max_comps]))
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
