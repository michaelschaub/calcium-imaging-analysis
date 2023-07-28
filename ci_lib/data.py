'''
This module contains the classes for handling decomposed data.
'''

import pathlib
import logging
import numpy as np
import pandas as pd
# pylint: disable-next=import-error
import h5py

from ci_lib.loading import reproducable_hash, load_h5, save_h5
from ci_lib.loading.alignment import align_spatials

LOGGER = logging.getLogger(__name__)


'''
Data objects loaded from files should be stored here by their hash digests
'''
LOADED_DATA = {}


class DecompData:
    '''
    The main class for handeling data decomposed into spatial and temporal components.
    Currently only supports data with trial information
    '''
    #TODO Add support for data without trial data and split DecompData into subclasses

    def __init__(self, data_frame, temporal_comps, spatial_comps, trial_starts, allowed_overlap=0,
                 cond_filter=None, trans_params=None, savefile=None, spatial_labels=None,
                 mean=None, stdev=None, dataset_id_column="dataset_id", logger=None):
        self.logger = LOGGER if logger is None else logger
        assert len(data_frame) == trial_starts.shape[0], (
            f"DataFrame data_frame and trial_starts do not have matching length \
                    ({len(data_frame)} != {len(trial_starts)})")
        self._df = data_frame.reset_index(drop=True)
        self._temps = temporal_comps
        if trans_params is None:
            self._spats = spatial_comps
        else:
            align_spatials(spatial_comps,trans_params, logger=self.logger)
        self._starts = trial_starts
        self._allowed_overlap = np.asarray(allowed_overlap)
        # has to be 0 if trials are not containing continous frames,
        # currently as 0 dim array cause save function doesn't handle ints yet

        self._spat_labels = spatial_labels
        self.dataset_id_column = dataset_id_column

        # Needed to calculate z-score based on mean and stdev over whole dataset
        # after splitting data into conditions
        self._mean = np.mean(self._temps,axis=0) if mean is None else mean
        self._stdev = np.std(self._temps,axis=0) if stdev is None else stdev


        if cond_filter is None:
            cond_filter = []
        self.conditions = cond_filter
        self._savefile = savefile

        if len(self) == 0:
            self.logger.warning("Created DecompData is empty")

    def copy(self):
        '''
        Return a (not deep) copy of this `DecompData` object.
        Usefull for then replacing some of the data with somthing different.
        '''
        return type(self)( self._df, self._temps, self._spats, self._starts,
                            allowed_overlap=self._allowed_overlap,
                            savefile=self.savefile, spatial_labels=self._spat_labels,
                            mean=self._mean, stdev=self._stdev, logger=self.logger )


    #Used for parcellations
    def recreate(self,temporal_comps=None, spatial_comps=None, spatial_labels=None):
        '''
        Create a copy of this `DecompData` object with some compoments replaced,
        depending on which parameter is given.
        '''
        # pylint: disable=protected-access
        data = self.copy()
        if temporal_comps is not None:
            data._temps = temporal_comps
            # Needed to calculate z-score based on mean and stdev over whole dataset
            # after splitting data into conditions
            data._mean = np.mean(data._temps,axis=0)
            data._stdev = np.std(data._temps,axis=0)
        if spatial_comps is not None:
            data._spats = spatial_comps
            data._spat_labels = spatial_labels
        data._savefile = None
        return data #TODO maybe as inplace instead?

    def concat(self, data, overwrite=False):
        ''' concats trials from List of DecompData to this DecompData'''
        if not isinstance(data, list):
            data = [data]
        if not overwrite:
            data = [self, *data]
        self._df = pd.concat([d._df for d in data], axis=0).reset_index(drop=True)
        time_offs = [0, *[ d.t_max for d in data ][:-1]]
        self._starts = np.concatenate([d.trial_starts + t for d,t in zip(data,time_offs)], axis=0)
        self._temps = np.concatenate([d.temporals_flat for d in data], axis=0)

    def save(self, file ):
        '''
        Save this DecompData object to file `file` as an h5
        '''
        LOGGER.info("%s has type %s", self._allowed_overlap, type(self._allowed_overlap))
        _ = save_h5( self, file, {"df"    : self._df,
                                        "temps" : self._temps,
                                        "spats" : self._spats,
                                        "starts" : self._starts,
                                        "overlap": self._allowed_overlap,
                                        "labels":self._spat_labels,
                                        "mean":self._mean,
                                        "stdev":self._stdev}, logger=self.logger)
        self._savefile = file

    @classmethod
    def load(cls, file, data_hash=None, try_loaded=False, logger=LOGGER):
        '''
        Load a saved DecompData object from an h5 file `file`
        '''
        # use already loaded data object if possible, may save a lot of memory
        if try_loaded and data_hash is not None and data_hash in LOADED_DATA:
            data = LOADED_DATA[data_hash]
        else:
            loaded = load_h5( file,
                              labels=["df", "temps", "spats", "starts","overlap","labels",
                                      "mean","stdev"],
                              logger=logger)
            # pylint: disable-next=unbalanced-tuple-unpacking
            _, data_frame, temps, spats, starts, allowed_overlap, spat_labels, mean, stdev = loaded
            data = cls(data_frame, temps, spats, starts, allowed_overlap=allowed_overlap,
                       spatial_labels=spat_labels, savefile=file, mean=mean, stdev=stdev,
                       logger=logger)
            LOADED_DATA[data.hash.digest()] = data
        return data

    @property
    def spatial_labels(self):
        '''The labels given to individual spatial components, None if none are given'''
        return self._spat_labels

    @property
    def hash(self):
        '''
        A hash calculated from the important components of this DecompData
        WARNING: MAY NOT WORK, STILL IN DEVELOPMENT
        '''
        return reproducable_hash(tuple( hsh.digest() for hsh in (
                            self.df_hash, self.temps_hash, self.spats_hash, self.starts_hash)))

    @property
    def df_hash(self):
        '''A hash calculated from the dataframe of this DecompData'''
        return reproducable_hash(self._df)

    @property
    def temps_hash(self):
        '''A hash calculated from the temporal components of this DecompData'''
        return reproducable_hash(self._temps)

    @property
    def spats_hash(self):
        '''A hash calculated from the spatial components of this DecompData'''
        return reproducable_hash(self._spats)

    @property
    def starts_hash(self):
        '''A hash calculated from the trial start times of this DecompData'''
        return reproducable_hash(self._starts)

    def check_hashes(self, hashes, warn=True ):
        '''Compares all hashes of this object with those given by `hashes`'''
        if reproducable_hash(tuple( hsh for hsh in hashes)).digest() == self.hash.digest():
            return True
        if warn:
            if hashes[0] is not None and hashes[0] != self.df_hash:
                self.logger.warning("df hashes do not match")
            if hashes[1] is not None and hashes[1] != self.temps_hash:
                self.logger.warning("temps hashes do not match")
            if hashes[2] is not None and hashes[2] != self.spats_hash:
                self.logger.warning("spats hashes do not match")
            if hashes[3] is not None and hashes[3] != self.starts_hash:
                self.logger.warning("starts hashes do not match")
        return False

    @property
    def savefile(self):
        '''
        Get path of the file, this DecompData object was last saved into or loaded from.
        returns None if object was never saved and not loaded or if attribute hashes in file
        do not match attribute hashes of object
        '''
        if (not self._savefile is None and pathlib.Path(self._savefile).is_file()):
            h5_file = h5py.File(self._savefile, "r")
            if self.check_hashes([ bytes.fromhex(h5_file[a].attrs["hash"])
                                    for a in ["df","temps","spats","starts"] ]):
                return self._savefile
        return None

    @property
    def n_components(self):
        '''The number of components the data is decomposed into'''
        return self._spats.shape[0]

    @property
    def n_xaxis(self):
        '''The width of the spatials in the x axis'''
        return self._spats.shape[1]

    @property
    def n_yaxis(self):
        '''The width of the spatials in the y axis'''
        return self._spats.shape[2]

    @property
    def t_max(self):
        '''The total length of temporal dimension'''
        return self._temps.shape[0]

    @property
    def trials_n(self):
        ''' The number of trials'''
        return self._df.shape[0]

    @property
    def frame(self):
        '''The pandas DataFrame containing the trial data'''
        return self._df

    @property
    def temporals_z_scored(self):
        '''
        Get z-score of temporal components of DecompData object, reshaped into trials.
        The new length of the individual trials is that of shortest trial.
        Z-score is calculated using the mean and standart deviation of the full dataset,
        even after splitting the DecompData-object into conditions.
        '''
        try:
            ends = np.roll(self._starts, -1)
            ends[-1] = self.t_max
            min_trial_len = np.min(ends - self._starts)
            temp_indx = np.array(np.arange(min_trial_len)[np.newaxis, :] + self._starts[:, np.newaxis], dtype=int)
            return self._temps[temp_indx] - self._mean * (1/self._stdev)
        except ValueError as err:
            if "cannot reshape array of size 0 into shape" in err.args[0]:
                return np.zeros((0, 0, self.n_components))
            raise

    @property
    def temporals(self):
        '''
        Get temporal components of DecompData object, reshaped into trials.
        The new length of the individual trials is that of shortest trial.
        '''
        try:
            ends = np.roll(self._starts, -1)
            ends[-1] = self.t_max
            min_trial_len = np.min(ends - self._starts)
            temp_indx = np.array(np.arange(min_trial_len)[np.newaxis, :] + self._starts[:, np.newaxis], dtype=int)
            return self._temps[temp_indx]
        except ValueError as err:
            if "cannot reshape array of size 0 into shape" in err.args[0]:
                return np.zeros((0, 0, self.n_components))
            raise

    @property
    def temporals_flat(self):
        '''
        Get temporal components of DecompData object, kept as one timeseries
        '''
        return self._temps

    @property
    def spatials(self):
        '''Get spatial components of DecompData object'''
        return self._spats

    @property
    def trial_starts(self):
        '''
        Get the trial start times of DecompData object
        '''
        return self._starts

    class PixelSlice:
        '''
        A helper class, designed to enable lazy slicing and accessing of recomposed pixel data
        by slicing spatials and temporals
        '''
        def __init__(self, temps, spats):
            self._temps = temps
            self._spats = spats

        def __getitem__(self, keys):
            if not isinstance(keys, tuple):
                keys = (keys,)
            temps = self._temps[keys[0], :]
            if len(keys) != 1:
                if len(keys) == 2:
                    spats = self._spats[:, keys[1]]
                else:
                    keys2 = keys[2] if len(keys) > 2 else slice(None, None, None)
                    spats = self._spats[:, keys[1], keys2]
            else:
                spats = self._spats
            return np.tensordot(temps, spats, (-1, 0))

        def __setitem__(self, keys, value):
            if np.any(value != 0):
                raise ValueError("value is not zero, PixelSlice.__setitem__ can only be used for masking")
            if not isinstance(keys, tuple):
                keys = (keys,)
            if np.all([k == slice(None) for k in keys[1:]]):
                self._temps[keys[0], :] = value
            elif keys[0] == slice(None):
                self._spats[(*keys[1:],)] = value
            else:
                raise ValueError("PixelSlice.__setitem__ can only mask temporal or spatial dimensions at once")

        @property
        def shape(self):
            return (self._temps.shape[0], *self._spats.shape[1:])

        @staticmethod
        def concat(pixels):
            return DecompData.PixelSlice.PixelConcat(pixels)

        class PixelConcat:
            def __init__(self, pixels):
                self._pixels = pixels
                self._stops  = np.cumsum([p._temps.shape[0] for p in pixels])
                self._temps  = np.split(np.arange(self._stops[-1]), self._stops[:-1])

            def __getitem__(self, keys):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                selected_t = np.arange(self._stops[-1])[keys[0]]
                temp_indx = [ts[np.isin(ts, selected_t)] for ts in self._temps]
                starts = (0,*self._stops[:-1])
                temp_indx = [ts - start for ts,start in zip(temp_indx,starts)]
                pixels = [pixel.__getitem__((temps, *keys[1:]))
                          for pixel, temps in zip(self._pixels, temp_indx)]
                return np.concatenate(pixels, axis=0)

            def __setitem__(self, keys, value):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                if np.all([k == slice(None) for k in keys[1:]]):
                    selected_t = np.arange(self._stops[-1])[keys[0]]
                    temp_indx = [ts[np.isin(ts, selected_t)] for ts in self._temps]
                    starts = (0,*self._stops[:-1])
                    temp_indx = [ts - start for ts,start in zip(temp_indx,starts)]
                    for pixel, temps in zip(self._pixels, temp_indx):
                        pixel.__setitem__(temps, value)
                elif keys[0] == slice(None):
                    for pixel in self._pixel:
                        pixel.__setitem__(keys, value)
                else:
                    raise ValueError("PixelConcat.__setitem__ can only mask temporal or spatial dimensions at once")

            @property
            def shape(self):
                return (self._stops[-1], *self._pixels[0].shape[1:])


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

    @property
    def pixel(self):
        '''
        Creates PixelSlice object, from which slices of recomposed pixel data can be accessed
        the first key is applied to the temporals, the second and third to the horizontal
        and vertical dimension of the spatials
        '''
        return DecompData.PixelSlice(self._temps, self._spats)

    def __getitem__(self, keys):
        '''
        Slices the DecompData object into a new DecompData object
        the first key is applied to the trial dimension, the second to the temporals by trial
        '''
        if not isinstance(keys, tuple):
            keys = (keys, slice(None, None, None))
        elif len(keys) < 2:
            keys = (keys[0], slice(None,None,None))
        try:
            data_frame = self._df.iloc[keys[0]]
        except NotImplementedError:
            data_frame = self._df.loc[keys[0]]
        spats = self._spats

        # TODO implement the resolution of key 1 more robustly!
        try:
            # if keys[1] is bool us it as mask on aranged array to create array of frames to keep
            assert np.array(keys[1]).dtype == bool
            trial_frames = np.array(np.arange(len(keys[1]))[keys[1]])
            #self.logger.debug(f"frames {trial_frames}")

        # especially this needs a better solution!
        # pylint: disable-next=bare-except
        except:
            try:
                # else use it to slice from aranged array
                #trial_frames = np.array(np.arange(np.max(np.diff(self._starts)))[keys[1]])
                # max can't work if we have some outliers in the frame length
                trial_frames = np.array(np.arange(
                                                np.min(np.diff(self._starts))+self._allowed_overlap
                                        )[keys[1]])
            except ValueError as err:
                if "zero-size array to reduction operation" in err.args[0]:
                    self.logger.warning("Data has size zero")
                    trial_frames = np.array([])
                else:
                    raise
        # starts of selected frames in old temps
        starts = np.array(self._starts[keys[0]])
        #self.logger.debug(f"starts {starts=}")

        # indices of temps in all selected frames (2d)
        selected_temps = np.array(trial_frames[np.newaxis, :] + starts[:, np.newaxis], dtype=int)
        #self.logger.debug(f"(frames + starts)={selected_temps}")

        # starts of selected frames in new temps
        if 0 == selected_temps.shape[1]:
            new_starts = np.zeros((selected_temps.shape[0],), dtype=int)
        elif 0 == selected_temps.shape[0]:
            new_starts = np.empty((0,), dtype=int)
        else:
            new_starts = np.insert(np.cumsum(np.diff(selected_temps[:-1, (0, -1)]) + 1), 0, 0)

        #self.logger.debug(f"{self._temps.shape=}")
        temps = self._temps[selected_temps.flatten()]

        try:
            data = DecompData(data_frame, temps, spats, new_starts,
                              spatial_labels=self._spat_labels,
                              mean=self._mean, stdev=self._stdev,
                              allowed_overlap=self._allowed_overlap)
        except AssertionError:
            self.logger.debug( starts.shape )
            self.logger.debug( selected_temps.shape )
            self.logger.debug( new_starts.shape )
            self.logger.debug( data_frame )
            self.logger.exception("Error in data.py")
            raise
        return data

    def __len__(self):
        return self._df.__len__()

    @property
    def conditions(self):
        '''Returns filtered conditions'''
        return self._conditional

    @conditions.setter
    def conditions(self, conditions):
        '''
        Get ConditionalData object created from self with conditions
        conditions can either be a list or dictionary of condition dictionaries
        '''
        self._cond_filters = conditions
        self._conditional = ConditionalData( self, conditions)

    @property
    def condition_filters(self):
        '''Returns conditions filter'''
        return self._cond_filters

    @condition_filters.setter  # just another interface
    def condition_filters(self, cond_filters):
        self.conditions = cond_filters

    def get_conditional(self, conditions):
        '''
        returns slice of self, containing all trials, where the trial data given
        by the keys of conditions match their corresponding values
        '''
        def check_attr(data_frame, attr, val):
            if callable(val):
                return val(getattr( data_frame, attr ))
            return getattr( data_frame, attr ) == val

        select = True
        for attr, cond_val in conditions.items():
            if isinstance(cond_val,list):
                any_matching = False
                for val in cond_val:
                    any_matching = any_matching | check_attr(self._df, attr, val)
                select = select & any_matching
            else:
                select = select & check_attr(self._df, attr, cond_val)
        #if(np.any_matching(select)):
        return self[select]
        #else:
            #return None

    def dataset_from_sessions(self, sessions, dataset_id):
        '''
        Creates a DecompData object, only containing specified sessions and sets its dataset_column
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
        data.frame[self.dataset_id_column] = dataset_id
        return session_data[0]

#TODO remove this class, it is really not necessary
class ConditionalData:
    '''
    This class represents a container of conditioned DecompData objects,
    keyed by the same keys as the conditions used to construct it
    '''

    def __init__( self, data, conditions ):
        if isinstance( conditions, dict ):
            self._data = { key : data.get_conditional(cond) for key, cond in conditions.items() }
        else:
            self._data = [ data.get_conditional(cond) for cond in conditions ]

    def keys(self):
        '''Returns the keys this ConditionalData is indexed by'''
        if isinstance( self._data, dict ):
            return self._data.keys()
        return range(len(self._data))

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys, slice(None, None, None))
        if isinstance( self._data, dict ):
            ret = self._data[keys[0]].__getitem__(keys[1:])
        else:
            dat = self._data[keys[0]]
            if isinstance(dat, DecompData ):
                ret = dat.__getitem__(keys[1:])
            else:
                ret = [ d.__getitem__(keys[1:]) for d in dat ]
        return ret


    def __getattr__(self, key):
        if isinstance( self._data, dict ):
            ret = { k : getattr(d, key) for k,d in self._data.items() }
        else:
            ret = [ getattr(d, key) for d in self._data ]
        return ret

    def __len__(self):
        return len(self._data)
