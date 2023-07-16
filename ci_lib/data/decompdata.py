'''Contains the `DecompData` class'''

import logging
from typing import TypedDict, Optional, Union
#from typing_extensions import Unpack
import numpy as np

Index = Union[int,slice,np.ndarray[bool]]
DecompDataList = Union['DecompData',list['DecompData']]
Array = np.ndarray
OptArray = Optional[Array]

LOGGER = logging.getLogger(__name__)

class DecompDataParams(TypedDict):
    '''kwargs dictionary for DecompData constructor'''
    spatial_labels: OptArray
    means: OptArray
    stdev: OptArray
    logger: Optional[logging.Logger]

class DecompData:
    '''The base class for handeling data decomposed into spatial and temporal components'''

    _temps       : Array
    _spats       : Array
    _spat_labels : Array

    # Needed to calculate z-score based on mean and stdev over whole dataset
    # after splitting data into conditions
    _mean  : Array
    _stdev : Array

    _logger : logging.Logger

    def __init__(self, temporal_comps : Array, spatial_comps : Array,
                 **kwargs : DecompDataParams):
                 #**kwargs : Unpack[DecompDataParams]):
        spatial_labels = kwargs.get("spatial_labels", None)
        mean           = kwargs.get("mean", None)
        stdev          = kwargs.get("stdev", None)
        self._logger   = kwargs.get("logger", LOGGER)

        n_t_comps = temporal_comps.shape[1]
        n_s_comps = spatial_comps.shape[0]
        assert n_t_comps == n_s_comps, (f"Number of components in temporals ({n_t_comps})"
                        f" does not match number of components in spatials ({n_s_comps})")
        if spatial_labels is not None:
            assert spatial_labels.shape[0] == n_s_comps, ("Number of labels"
                        f" does not match number of components in spatials ({n_s_comps})")

        self._temps = temporal_comps
        self._spats = spatial_comps
        self._spat_labels = spatial_labels

        self._mean = np.mean(self._temps,axis=0) if mean is None else mean
        self._stdev = np.std(self._temps,axis=0) if stdev is None else stdev

### PROPERTIES

    @property
    def temporals(self) -> Array:
        '''Get temporal components of DecompData object, kept as one timeseries'''
        return self._temps

    @property
    def temporals_z_scored(self):
        '''
        Get z-score of temporal components of DecompData object, kept as one timeseries.
        Z-score is calculated using the mean and standart deviation of the full dataset,
        even after splitting the DecompData-object into conditions.
        '''
        return self._temps - self._mean * (1/self._stdev)

    @property
    def spatials(self) -> Array:
        '''Get spatial components of DecompData object'''
        return self._spats

    @property
    def spatial_labels(self) -> Array:
        '''The labels given to individual spatial components, None if none are given'''
        return self._spat_labels

    @property
    def n_components(self) -> int:
        '''The number of components the data is decomposed into'''
        return self._spats.shape[0]

    @property
    def n_xaxis(self) -> int:
        '''The width of the spatials in the x axis'''
        return self._spats.shape[1]

    @property
    def n_yaxis(self) -> int:
        '''The width of the spatials in the y axis'''
        return self._spats.shape[2]

    @property
    def t_max(self) -> int:
        '''The total length of temporal dimension'''
        return self._temps.shape[0]

### FUNCTIONS

    def copy(self, temporal_comps: OptArray = None,
                    spatial_comps:  OptArray = None,
                    spatial_labels: OptArray = None) -> 'DecompData':
        '''Create a copy of this `DecompData` object optionally with some/all compoments replaced'''
        if temporal_comps is None:
            temporal_comps = np.copy(self._temps)
            mean = np.copy(self._mean)
            stdev = np.copy(self._stdev)
        else:
            mean = np.mean(temporal_comps,axis=0)
            stdev = np.std(temporal_comps,axis=0)
        if spatial_comps is None:
            spatial_comps  = np.copy(self._spats)
        if spatial_labels is None:
            spatial_labels = np.copy(self._spat_labels)
        return DecompData(temporal_comps, spatial_comps, spatial_labels=spatial_labels,
                          mean=mean, stdev=stdev, logger=self._logger)

    def concat(self, data:DecompDataList, overwrite:bool=False) -> 'DecompData':
        '''
        Concats trials from list of `DecompData` to this `DecompData`
        WARNING: The `DecompData` objects in `data` should be considered modified and invalid after
        '''
        if not isinstance(data, list):
            data = [data]
        if not overwrite:
            data = [self, *data]
        if len(data) > 1:
            matching_spatials = (data[0].spatials==np.array(data[1:])).all(axis=0)
            assert matching_spatials, "Spatials do not seem to match"
        self._temps = np.concatenate([d.temporals for d in data], axis=0)
        return self

    def __getitem__(self, key:Index) -> 'DecompData':
        '''
        Slices the DecompData object into a new DecompData object
        the first key is applied to the temporal dimension
        '''
        return self.copy(temporal_comps=self._temps[key])

    #TODO implement pixel, save and load
