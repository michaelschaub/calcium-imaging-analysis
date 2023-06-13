import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type
from ci_lib import DecompData


def calc_means(temps):
    return np.mean(temps, axis=1) # average over frames


class Means(Features):
    _type = Feature_Type.NODE

    def __init__(self, frame, data, feature, file=None, time_resolved=False, full=False):
        super().__init__(frame=frame, data=data, feature=feature, file=file, full=full)
        self._time_resolved = time_resolved #only needed cause it's not properly saved


    @staticmethod
    def create(data, max_comps=None, logger=LOGGER, window=None, start=None, stop=None,full=False,z_scored=True): #TODO z_score default ot False
        if max_comps is not None:
            logger.warn("DEPRECATED: max_comps parameter in features can not garanty sensible choice of components, use n_components parameter for parcellations instead")
        if window is None:
            feat = Means(data.frame, data, feature=calc_means(data.temporals[:, slice(start,stop), :max_comps])[:,np.newaxis,:])  #TODO start:stop should be supported by window as well
        else:
            trials , phase_length, comps  =   data.temporals[:, slice(start,stop), :max_comps].shape
            windows = [range(i,i+window) for i in range(0,phase_length-window+1)]

            feat_val = np.zeros((trials,len(windows),comps if max_comps is None else max_comps))
            for w,window in enumerate(windows):
                feat_val[:,w,:] = calc_means(data.temporals[:, slice(start,stop), :][:, window, :max_comps] if not z_scored else data.temporals_z_scored[:, slice(start,stop), :][:, window, :max_comps])

                
            feat = Means(data.frame, data, feature=feat_val, time_resolved=True,full=full)

        return feat

    def flatten(self, timepoints=slice(None), feat=None):
        if feat is None:
            feat = self._feature

        feat = feat[:,timepoints,:]
        mask = np.ones((feat.shape[1:]), dtype=bool)
        return feat[:,mask]


    @property
    def pixel(self):
        return DecompData.PixelSlice(self._feature, self.data.spatials[:self._feature.shape[1]])

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
