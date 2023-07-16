'''Contains the `TrialData` class'''

import logging
from typing import Optional, Union, Any
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

MultiIndex = tuple
Index = Union[np.ndarray[bool], int, slice, np.ndarray[int], MultiIndex, np.ndarray[MultiIndex]]
TrialDataList = Union['TrialData',list['TrialData']]
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

### PROPERTIES

    @property
    def n_trials(self) -> int:
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

### FUNCTIONS

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

    def concat(self, data:TrialDataList, overwrite:bool=False) -> 'TrialData':
        '''
        Concats trials from list of `TrialData` to this `TrialData`
        WARNING: The `TrialData` objects in `data` should be considered modified and invalid after
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
        return self

    def _events_by_indx(self, key):
        #TODO add event kwargs
        return {evnt_name : (evnt_indx[0][key],) for evnt_name, evnt_indx in self._events.items()}

    def __getitem__(self, key:Index) -> 'TrialData':
        '''
        Slices the `TrialData` object into a new `TrialData` object
        If `key` is an integer, integer array/series or slice, trials will be selected \
by dataframe integer position
        If `key` is a tuple, tuple array/series or slice, trials will be selected via multiindex
        If `key` is a bool array, ; it must have length `n_trials`
        If `key` is a bool series, ; it must have the same index as `dataframe`
        '''

        def cmp_slice_type(slc, typ):
            '''Small helper function to compare the type of a slice to `t`'''
            return ( (slc.start is None or isinstance(slc.start, typ))
                and (slc.stop is not None and not isinstance(slc.start, typ)))

        if isinstance(slice):
            if cmp_slice_type(key, int):
                dataframe = self._dataframe.iloc[key]
                events = self._events_by_indx(key)
            elif cmp_slice_type(key, tuple):
                dataframe = self._dataframe.loc[key]
                #TODO add event creation
            else:
                raise ValueError

        # Treat `tuple` seperatly, since `pd.Series` would convert it to multiple singular keys
        if isinstance(tuple):
            key = pd.Series([key])
        # Convert all other key(-arrays) into a `pandas.Series`
        if not isinstance(key, pd.Series):
            key = pd.Series(key)
        # `key` is now a `pandas.Series`, determine accessing method from `key.dtype`
        if key.dtype == int:
            dataframe = self._dataframe.iloc[key]
            events = self._events_by_indx(key)
        elif key.dtype == tuple:
            dataframe = self._dataframe.loc[key]
            #TODO add event creation
        elif key.dtype == bool:
            assert len(key) == len(self._dataframe), ""
            dataframe = self._dataframe.loc[key]
            events = self._events_by_indx(key)
        else:
            raise ValueError

        return TrialData(dataframe, events)

    def get_conditional(self, conditions:dict) -> 'TrialData':
        '''
        returns slice of self, containing all trials, where the trial data given
        by the keys of conditions match their corresponding values
        '''
        def check_attr(data_frame, attr, val):
            if callable(val):
                return val(getattr( data_frame, attr ))
            return getattr( data_frame, attr ) == val

        select = pd.Series(True, index=self._dataframe.index)
        for attr, cond_val in conditions.items():
            if isinstance(cond_val,list):
                any_matching = pd.Series(False, index=self._dataframe.index)
                for val in cond_val:
                    any_matching = any_matching | check_attr(self._dataframe, attr, val)
                select = select & any_matching
            else:
                select = select & check_attr(self._dataframe, attr, cond_val)
        return self[select]

    def dataset_from_sessions(self, sessions, dataset_id):
        '''
        Creates a TrialData object, only containing specified sessions and sets its dataset_column
        to the specified dataset_id
        sessions should be a list of dicts, each containing a `subject_id` string and a `datetime`
        `numpy.datetime64` value
        '''
        subject_ids = [s['subject_id'] for s in sessions]
        dates = [s['datetime'] for s in sessions]
        def date_compare(date, date_comp):
            return np.array(date, dtype=date.dtype) == date_comp
        session_data = [self.get_conditional({'subject_id': subject_id})
                                                    for subject_id in subject_ids]
        session_data = [data.get_conditional({'date_time' : (
                                lambda d, d_comp=date: date_compare(d,d_comp)) })
                                                    for date, data in zip(dates, session_data)]
        data = session_data[0]
        data.concat(session_data, overwrite=True)
        data.frame['dataset_id'] = dataset_id
        return session_data[0]

    def subsample(self,size,seed=None):
        '''
        Subsampling Trials to balance number of datapoints between different conditions
        :param size: Number of trials to sample
        :param seed: The seed for the rng
        '''
        rng = np.random.default_rng(seed)
        select_n = rng.choice(self.trials_n,size=size,replace=False)
        data = self[select_n]
        return data
