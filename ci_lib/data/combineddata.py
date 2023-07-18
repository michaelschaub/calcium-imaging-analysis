'''Contains the `CombinedData` class'''

import logging
from typing import TypedDict, Optional
from typing_extensions import Unpack
import pandas as pd

from .decompdata import DecompData, Array, OptArray, LOGGER as DECOMP_LOGGER
from .trialdata import TrialData, Events

LOGGER = logging.getLogger(__name__)

class CombinedDataParams(TypedDict):
    '''kwargs dictionary for CombinedData constructor'''
    spatial_labels: OptArray
    means: OptArray
    stdev: OptArray
    logger: Optional[logging.Logger]

class CombinedData(DecompData, TrialData):
    '''The standard class for handeling data decomposed data with trial data'''

    def __init__(self, dataframe : pd.DataFrame, events : Events,
                 temporal_comps : Array, spatial_comps : Array,
                 **kwargs : Unpack[CombinedDataParams]):
        logger = kwargs.pop('logger', LOGGER)
        DecompData.__init__(self, temporal_comps, spatial_comps, logger=logger, **kwargs)
        if events['trials'][0].mask[-1,1]:
            events['trials'][0][-1,1] = self.t_max
        TrialData.__init__(self, dataframe, logger=logger, **events)

    @classmethod
    def from_trial_and_decomp_data(cls, trials : TrialData, decomp : DecompData):
        '''Construct a `CombinedData` object from `TrialData` and `DecompData`'''
        return CombinedData(trials.dataframe, trials.events,
                            decomp.temporals, decomp.spatials, spatial_labels=decomp.spatial_labels,
                            mean=decomp.mean,stdev=decomp.stdev,
                            logger=decomp.logger if not decomp.logger==DECOMP_LOGGER else LOGGER)

### PROPERTIES

    @property
    def temporals(self):
        '''
        Get temporal components of `CombinedData` object, reshaped into trials.
        The new length of the individual trials is that of shortest trial.
        '''
        #TODO

    @property
    def temporals_flat(self):
        '''
        Get temporal components of `CombinedData` object, kept as one timeseries
        '''
        return DecompData.temporals(self)

### FUNCTIONS

    def copy(self, temporal_comps: OptArray = None,
                    spatial_comps:  OptArray = None,
                    spatial_labels: OptArray = None) -> 'CombinedData':
        '''
        Create a copy of this `CombinedData` object optionally with some/all compoments replaced
        '''
        decomp = DecompData.copy(self, temporal_comps, spatial_comps, spatial_labels)
        trials = TrialData.copy(self)
        return CombinedData.from_trial_and_decomp_data(decomp, trials)

    def __getitem__(self, keys):
        '''
        Slices the `CombinedData` object into a new `CombinedData` object
        the first key is applied to the trial dimension, the second to the temporals by trial
        '''
        #TODO
