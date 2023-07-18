'''Contains the `TrialData` class'''

import logging
from typing import Optional, Union, Any
from typing_extensions import Unpack
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


EventIndex = Union[list[int], list[np.ndarray[int]]]
EventArgs = dict[str,list[Any]]

class Event:
    '''
    A class holding a the time indices and optional keyworded extra arguments of some event
    indices and kwargs are lists/arrays of length n_trials
    '''

    _indx:        np.ndarray[int]
    _indx_ends:   Optional[np.ndarray[int]]            = None
    _kwargs:      dict[str, np.ndarray[Any]]
    _kwargs_ends: dict[str, Optional[np.ndarray[int]]]

    def __init__(self, indices : EventIndex, **kwargs:Unpack[EventArgs]):
        try:
            self._indx = np.array(indices, dtype=int)
            if len(self._indx.shape) == 1:
                self._indx = self._indx.reshape((self._indx.shape[0], 1))
        except ValueError:
            self._indx = np.array(np.concatenate(indices), dtype=int)
            self._indx_ends = np.cumsum([len(i) for i in indices])[:-1]

        self._kwargs      = {}
        self._kwargs_ends = {}
        for key, arg in kwargs.items():
            assert len(arg) == len(indices), (f"length of Event arg \"{key}\" "
                                               "does not match length of indices")
            try:
                self._kwargs[key] = np.array(arg)
                self._kwargs_ends[key] = None
            except ValueError:
                self._kwargs[key] = np.concatenate(arg)
                self._kwargs_ends[key] = np.cumsum([len(a) for a in arg])[:-1]

    @property
    def indices(self) -> Union[np.ndarray[int], np.ndarray[np.ndarray[int]]]:
        '''Returns the array/list of arrays containing the event's indices'''
        if self._indx_ends is None:
            return self._indx
        return np.array(np.split(self._indx, self._indx_ends), dtype=object)

    @property
    def keys(self) -> set[str]:
        '''Returns keys for kwargs'''
        return set(self._kwargs.keys)

    @property
    def kwargs(self) -> EventArgs:
        '''Returns all event arguments in a dictionary'''
        return {key : self.kwarg(key) for key in self._kwargs}

    def kwarg(self, key : str) -> Union[np.ndarray[Any], np.ndarray[np.ndarray[Any]]]:
        '''Returns the array/list of arrays given by argument `key`'''
        if self._kwargs_ends[key] is None:
            return self._kwargs[key]
        return np.array(np.split(self._kwargs[key], self._kwargs_ends[key]), dtype=object)

    def copy(self):
        '''Return a copy of this `Event` object'''
        return Event(self.indices.copy(), **{key : arg.copy() for key,arg in self.kwargs.items()})

    def __getattr__(self, key : str) -> Union[np.ndarray[Any], list[np.ndarray[Any]]]:
        return self.kwarg(key)

    def __getitem__(self, key : Union[int, slice, list[bool]]) -> 'Event':
        if np.isscalar(key):
            return Event([self.indices[key]], **{k : [arg[key]] for k, arg in self.kwargs.items()})
        return Event(self.indices[key], **{k : arg[key] for k, arg in self.kwargs.items()})

    def __len__(self):
        return self._indx.shape[0] if self._indx_ends is None else self._indx_ends.shape[0]+1

    def __repr__(self):
        return repr((self.indices, self.kwargs))

    def __str__(self):
        return str((self.indices, self.kwargs))

    def __iadd__(self, offset):
        n_neg = self._indx >= 0
        if np.isscalar(offset):
            self._indx[n_neg] += offset
        elif self._indx_ends is None:
            self._indx[n_neg] += offset[:,None][n_neg]
        else:
            lengths = np.diff([0, *self._indx_ends, len(self._indx)])
            offset_trialed = [offs * np.ones(l, dtype=int)
                              for offs, l in zip(offset, lengths)]
            self._indx[n_neg] += np.concatenate(offset_trialed)[n_neg]
        return self

    def __add__(self, offset):
        event = self.copy()
        event += offset
        return event

    @staticmethod
    def concat(*events):
        '''Concats events into a single event'''
        indices = [indx for e in events for indx in e.indices]
        kwargs = [e.kwargs for e in events]
        kwargs = {key : [arg for kw in kwargs for arg in kw[key]] for key in kwargs[0]}
        return Event(indices, **kwargs)

Events = dict[str, Union[Event, tuple[EventIndex, EventArgs]]]

MultiIndex = tuple
TrialIndex = Union[np.ndarray[bool], slice,
                   int, np.ndarray[int],
                   MultiIndex, np.ndarray[MultiIndex]]
TrialDataList = Union['TrialData', list['TrialData']]


def convert_trial_indx_tuple(ind, incl_zero:bool) -> np.ndarray[int]:
    '''
    Convert ind into a `numpy.ndarray` a, with zero=a[0], first=a[-2], last=a[-1]
    If no `incl_zero`, it contains [first, last], with zero being equal to first
    If `incl_bool`, it contains [zero, first, last]
    '''
    if not isinstance(ind, np.ndarray):
        ind = np.array(ind, ndmin=1)
    assert len(ind)<=3, ("Trial indices cannot contain more than 3 values "
                        f"(violated by {ind})")
    if len(ind) == 1:
        ind = np.array([ind[0], ind[0], -1])
    elif len(ind) == 2:
        ind = np.array([ind[0], ind[0], ind[1]])
    if not incl_zero:
        return np.array(ind[1:], dtype=int)
    return ind

def trial_index_format(indices) -> np.ndarray[int]:
    '''Converts a 2d list of trial indices to proper formating and fills trial stops'''
    # trial indices have to be formated into a np.array, containing meaningful starts and ends,
    # if trial indices do not contain a (optional) trial zero, first and last index
    if (not isinstance(indices, np.ndarray) or len(indices.shape) < 2
        or indices.shape[1] not in (2,3)):

        incl_zero = 3 == max(*[len(ind) for ind in indices])
        indices = np.array([convert_trial_indx_tuple(ind, incl_zero) for ind in indices])
    # replace not determined ends (==-1) with start of next trial
    for i in np.where(indices[:,-1] == -1)[0]:
        # not possible for last trial
        if i < len(indices) - 1:
            indices[i,-1] = indices[i+1,-2]
    zeros = indices[:,0]
    starts = indices[:,-2]
    stops = indices[:,-1]
    assert np.logical_and(starts <= zeros, np.logical_or(stops == -1, zeros < stops)).all(), (
            "Trial zero indices have to be between starts and stops")
    return np.array(indices, dtype=int)

class TrialData:
    '''The base class for handeling trial data as dataframe and eventlists'''

    _dataframe : pd.DataFrame
    _events    : Events
    _trial     : Event

    logger : logging.Logger

    def __init__(self, dataframe : pd.DataFrame, logger:logging.Logger=LOGGER,
                 **events : Unpack[Events]):
        self._dataframe = dataframe.reset_index(drop=True)
        self.logger     = logger

        n_trials = self._dataframe.shape[0]

        self._events = {}
        for evnt_name, event in events.items():
            if not isinstance(event, Event):
                if isinstance(event, tuple):
                    event = Event(event[0], **event[1])
                else:
                    event = Event(event)
            assert len(event) == n_trials, (f'Event "{evnt_name}" length does not '
                                             'match length of dataframe')
            self._events[evnt_name] = event

        assert "trial" in self._events.keys(), ("`TrialData` `events` needs to contain "
                                                "a \"trial\" event.")
        indices = trial_index_format(self._events["trial"].indices)
        self._trial = Event(indices, **self._events["trial"].kwargs)
        self._events["trial"] = self._trial

### PROPERTIES

    @property
    def n_trials(self) -> int:
        ''' The number of trials'''
        return self._dataframe.shape[0]

    @property
    def dataframe(self) -> pd.DataFrame:
        '''The pandas DataFrame containing the trial data'''
        return self._dataframe

    @property
    def events(self) -> Events:
        '''The dictionary containing the trial events'''
        return self._events

    @property
    def trials(self) -> np.ndarray[int]:
        '''The time indices to which trial times are relative'''
        return self._trial.indices[:,0]

    @property
    def starts(self) -> np.ndarray[int]:
        '''The time indices defining the start of a trial, relevant for slicing'''
        return self._trial.indices[:,-2]

    @property
    def stops(self) -> np.ndarray[int]:
        '''The time indices defining the stop of a trial, relevant for slicing'''
        return self._trial.indices[:,-1]

### FUNCTIONS

    def __len__(self):
        return self.n_trials

    def copy(self) -> 'TrialData':
        '''Return a copy of this `TrialData` object'''
        return TrialData(self._dataframe.copy(), logger=self.logger,
                         **{key : arg.copy() for key, arg in self._events.items()})

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
        trial_ends = np.array([d.stops[-1] for d in data])

        if (trial_ends[:-1] == -1).any():
            raise ValueError("Concat can not be performed with arrays"
                             " containing trials without definite boundaries"
                             " (ends = -1 or only starts where given)")

        events = {e_name : [dat.events[e_name].copy() for dat in data]
                  for e_name, e in self.events.items()}
        for trial, offset in enumerate(trial_ends[:-1]):
            for e_name in self.events:
                events[e_name][trial+1] += offset
        self._events = {e_name : Event.concat(*events[e_name]) for e_name in self.events}

        trial_indices = trial_index_format(self._events["trial"].indices)
        self._trial = Event(trial_indices, **self._events["trial"].kwargs)
        self._events["trial"] = self._trial
        return self

    def _events_by_indx(self, key):
        #TODO add event kwargs
        return {evnt_name : (evnt_indx[0][key],) for evnt_name, evnt_indx in self._events.items()}

    def __getitem__(self, key:TrialIndex) -> 'TrialData':
        '''
        Slices the `TrialData` object into a new `TrialData` object
        If `key` is an integer, integer array/series or slice, trials will be selected \
by dataframe integer position
        If `key` is a tuple, tuple array/series or slice, trials will be selected via multiindex
        If `key` is a bool array, ; it must have length `n_trials`
        If `key` is a bool series, ; it must have the same index as `dataframe`
        '''

        def cmp_slice_type(slc, typ):
            '''Small helper function to compare the type of a slice to `typ`'''
            return ( (slc.start is None or isinstance(slc.start, typ))
                and (slc.stop is None or isinstance(slc.start, typ)))

        if isinstance(key, slice):
            if cmp_slice_type(key, int):
                dataframe = self._dataframe.iloc[key]
                events = {e_name : event[key] for e_name, event in self._events.items()}
            elif cmp_slice_type(key, tuple):
                dataframe = self._dataframe.loc[key]
                #TODO add event creation
                raise NotImplementedError("Handling multiindices is not yet implemented")
            else:
                raise ValueError("The key slice should contain only int or only tuple")
        else:
            # Treat `tuple` seperatly, since `pd.Series` would convert it to multiple singular keys
            if isinstance(key, tuple):
                key = pd.Series([key])
            elif isinstance(key, int):
                key = pd.Series([key])
            # Convert all other key(-arrays) into a `pandas.Series`
            if not isinstance(key, pd.Series):
                key = pd.Series(key)
            # `key` is now a `pandas.Series`, determine accessing method from `key.dtype`
            if key.dtype == int:
                dataframe = self._dataframe.iloc[key]
                events = {e_name : event[key] for e_name, event in self._events.items()}
            elif key.dtype == tuple:
                dataframe = self._dataframe.loc[key]
                #TODO add event creation
                raise NotImplementedError("Handling multiindices is not yet implemented")
            elif key.dtype == bool:
                assert len(key) == len(self._dataframe), ""
                dataframe = self._dataframe.loc[key]
                events = {e_name : event[key] for e_name, event in self._events.items()}
            else:
                raise ValueError("key dtype must be slice, "
                                 "(array/Series of) int or (array/Series of) tuple")

        return TrialData(dataframe, logger=self.logger, **events)

    # TODO add conditions affecting events and event params
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
        data = session_data[0].concat(session_data, overwrite=True)
        data.frame['dataset_id'] = dataset_id
        return session_data[0]

    def subsample(self,size,seed=None):
        '''
        Subsampling Trials to balance number of datapoints between different conditions
        :param size: Number of trials to sample
        :param seed: The seed for the rng
        '''
        if (self.stops == -1).any():
            raise ValueError("subsample can not be performed with arrays"
                             " containing trials without definite boundaries"
                             " (ends = -1 or only starts where given)")
        rng = np.random.default_rng(seed)
        select_n = rng.choice(self.n_trials,size=size,replace=False)
        data = self[select_n]
        return data
