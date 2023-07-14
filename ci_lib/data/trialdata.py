'''Contains the `TrialData` class'''

import logging
from typing import Optional, Union, Any
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

Event = tuple[
        Union[np.ndarray[int],np.ma.MaskedArray[int],np.ndarray[slice]],
        Optional[dict[str,np.ndarray[Any]]]
        ]
Events = dict[str, Event]

class TrialData:
    '''The base class for handeling trial data as dataframe and eventlists'''

    _dataframe : pd.DataFrame
    _events    : Events

    _logger : logging.Logger

    def __init__(self, dataframe : pd.DataFrame, events : Events, logger:logging.Logger=LOGGER):
        trials_n = self._dataframe.shape[0]
        for evnt_name, event in events.copy().items():
            evnt_indx = event[0]
            evnt_indx_correct = ((len(evnt_indx.shape)==2 and event.shape[1] == 2)
                                or len(evnt_indx.shape)==1 and event.dtype in [int, slice])
            assert evnt_indx_correct, f"Event {evnt_name} has a malformed temporal index"
            assert evnt_indx.shape[0]==trials_n, f"Event {evnt_name} has a malformed temporal index"
            if len(evnt_indx.shape)==1:
                # Convert the event's temporal indices to (masked) [start,stop] arrays
                if event.dtype == int:
                    # Only the trial starts are given, infer stops from next start
                    evnt_ends = np.roll(evnt_indx, -1)
                    evnt_slcs = [[b, e] for b,e in zip(evnt_indx, evnt_ends)]
                    evnt_slcs[-1][1] = -1
                    events[evnt_name][0] = np.ma.array(evnt_slcs, dtype=int,
                                                     mask=np.array(evnt_slcs)<0)
                elif event.dtype == slice:
                    # Indices are given as slices, convert to masked array
                    evnt_slcs = np.array([[s.start, s.stop] for s in evnt_indx], dtype=object)
                    events[evnt_name][0] = np.ma.array(evnt_slcs, dtype=int,
                                                    #pylint: disable-next=C0121
                                                     mask=np.array(evnt_slcs)==None)

            #TODO add check for the optional event kwargs
        assert ("trial" in events.keys), "`TrialData` `events` needs to contain a \"trial\" event."

        self._dataframe = dataframe.reset_index(drop=True)
        self._events    = events
        self._logger    = logger

    def copy(self) -> 'TrialData':
        '''Return a copy of this `TrialData` object'''
        #TODO add deep copy of events
        return TrialData(self._dataframe.copy(), self._events, self._logger)

    def offset_events(self, offset:int):
        '''Add a constant offset to all event indices'''
        for event_indx in [e[0] for e in self._events.values()]:
            if event_indx.dtype == int:
                event_indx += offset
            else:
                # else event_index should contain slices
                event_indx_shape = event_indx.shape
                event_indx = np.array([slice(slc.first + offset, slc.second + offset)
                                        for slc in event_indx.flatten()]).reshape(event_indx_shape)

    def concat(self, data:Union['TrialData',list['TrialData']], overwrite:bool=False):
        '''
        concats trials from list of `TrialData` to this `TrialData`
        the `TrialData` objects in `data` should be considered modified and invalid after
        '''
        if not isinstance(data, list):
            data = [data]
        if not overwrite:
            data = [self, *data]
        self._dataframe = pd.concat([d.dataframe for d in data], axis=0).reset_index(drop=True)
        trial_ends = np.ma.array([d.events['trial'] for d in data])[:,:,1]

        if trial_ends.mask.any():
            raise ValueError("Concat can not be performed with arrays"
                             " containing trials without definite sizes"
                             " (slices with `None` or only starts where given)")

        for dat, offset in range(data[1:], trial_ends[:-1]):
            dat.offset_events(offset)

        dat_evnts = [dat.events for dat in data]
        evnt_keys = set(key for key in evnt.keys() for evnt in dat_evnts)
        #TODO add treating of event kwargs
        self._events = { e_name : (np.concatenate(evnts[e_name][0] for evnts in dat_evnts),)
                        for e_name in evnt_keys}

    @property
    def trials_n(self) -> int:
        ''' The number of trials'''
        return self._dataframe.shape[0]

    @property
    def frame(self) -> pd.DataFrame:
        '''The pandas DataFrame containing the trial data'''
        return self._dataframe

    @property
    def events(self) -> Events:
        '''The dictionary containing the trial events'''
        return self._events
