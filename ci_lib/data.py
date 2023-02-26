from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import h5py
import pathlib
import logging
LOGGER = logging.getLogger(__name__)


from ci_lib.loading import reproducable_hash, load_h5, save_h5
from ci_lib.loading.alignment import align_spatials

class Data(ABC):
    @abstractmethod
    def load(self, file, **kwargs):
        pass

    def binary_operation( a, b ):
        notNone = lambda x,y : y if x is None else x
        try:
            a_df, a_temps, a_spats, a_starts, a_labels = a._op_data(b)
        except AttributeError:
            a_df = None
            a_temps = a
            a_spats = None
            a_starts = None
            a_labels = None
        try:
            b_df, b_temps, b_spats, b_starts, b_labels = b._op_data(a)
        except AttributeError:
            b_df = None
            b_temps = b
            b_spats = None
            b_starts = None
            b_labels = None
        return a_temps, b_temps, notNone(a_df,b_df), notNone(a_spats, b_spats), notNone(a_starts, b_starts), notNone(a_labels,b_labels)

    '''
    Data objects loaded from files should be stored here by their hash digests
    '''
    LOADED_DATA = {}


class DecompData(Data):
    def __init__(self, df, temporal_comps, spatial_comps, trial_starts, allowed_overlap=0, cond_filter=None, trans_params=None, savefile=None, spatial_labels=None, mean=None, stdev=None, dataset_id_column="dataset_id", logger=None):
        self.logger = LOGGER if logger is None else logger
        #TODO remove first check
        assert len(df) != trial_starts.shape[0]-1, (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})\n\t(maybe remove last entry?)")

        assert len(df) == trial_starts.shape[0], (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})")
        self._df = df
        self._temps = temporal_comps
        self._spats = spatial_comps if trans_params is None else align_spatials(spatial_comps,trans_params, logger=self.logger)
        self._starts = trial_starts
        self._allowed_overlap = np.asarray(allowed_overlap) #has to be 0 if trials are not containing continous frames, currently as 0 dim array cause save function doesn't handle ints yet

        self._spat_labels = spatial_labels
        self.dataset_id_column = dataset_id_column

        #Needed to calculate z-score based on mean and stdev over whole dataset after splitting data into conditions
        self._mean = np.mean(self._temps,axis=0) if mean is None else mean
        self._stdev = np.std(self._temps,axis=0) if stdev is None else stdev


        if cond_filter is None:
            cond_filter = []
        self.conditions = cond_filter
        self._savefile = savefile

        if len(self) == 0:
            self.logger.warning("Created DecompData is empty")

    def copy(self):
        return type(self)( self._df, self._temps, self._spats, self._starts, allowed_overlap=self._allowed_overlap,
                            savefile=self.savefile, spatial_labels=self._spat_labels,
                            mean=self._mean, stdev=self._stdev, logger=self.logger )


    #Used for parcellations
    def recreate(self,temporal_comps=None, spatial_comps=None, spatial_labels=None):
        data = self.copy()
        if temporal_comps is not None:
            data._temps = temporal_comps
            #Needed to calculate z-score based on mean and stdev over whole dataset after splitting data into conditions
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
        self._df = pd.concat([d._df for d in data], axis=0)
        time_offs = [0, *[ d.t_max for d in data ][:-1]]
        self._starts = np.concatenate([d._starts + t for d,t in zip(data,time_offs)], axis=0)
        self._temps = np.concatenate([d._temps for d in data], axis=0)

    def save(self, file ):
        LOGGER.info(f"{self._allowed_overlap} has type {type(self._allowed_overlap)}")
        h5_file = save_h5( self, file, {"df"    : self._df,
                                        "temps" : self._temps,
                                        "spats" : self._spats,
                                        "starts" : self._starts,
                                        "overlap": self._allowed_overlap,
                                        "labels":self._spat_labels,
                                        "mean":self._mean,
                                        "stdev":self._stdev}, logger=self.logger)
        self._savefile = file

    @classmethod
    def load(Class, file, data_hash=None, try_loaded=False, logger=LOGGER):
        # use already loaded data object if possible, may save a lot of memory
        if try_loaded and data_hash is not None and data_hash in Data.LOADED_DATA:
            data = Data.LOADED_DATA[data_hash]
        else:
            _, df, temps, spats, starts, allowed_overlap, spat_labels, mean, stdev = load_h5( file, labels=["df", "temps", "spats", "starts","overlap","labels","mean","stdev"], logger=logger)
            data = Class(df, temps, spats, starts, allowed_overlap=allowed_overlap, spatial_labels=spat_labels, savefile=file, mean=mean, stdev=stdev, logger=logger)
            Data.LOADED_DATA[data.hash.digest()] = data
        return data

    @property
    def spatial_labels(self):
        return self._spat_labels

    @property
    def hash(self):
        return reproducable_hash(tuple( hsh.digest() for hsh in (self.df_hash, self.temps_hash, self.spats_hash, self.starts_hash)))

    @property
    def df_hash(self):
        return reproducable_hash(self._df)

    @property
    def temps_hash(self):
        return reproducable_hash(self._temps)

    @property
    def spats_hash(self):
        return reproducable_hash(self._spats)

    @property
    def starts_hash(self):
        return reproducable_hash(self._starts)

    def check_hashes(self, hashes, warn=True ):
        if reproducable_hash(tuple( hsh for hsh in hashes)).digest() == self.hash.digest():
            return True
        elif warn:
            if hashes[0] is not None and hashes[0] != self.df_hash:
                self.logger.warn("df hashes do not match")
            if hashes[1] is not None and hashes[1] != self.temps_hash:
                self.logger.warn("temps hashes do not match")
            if hashes[2] is not None and hashes[2] != self.spats_hash:
                self.logger.warn("spats hashes do not match")
            if hashes[3] is not None and hashes[3] != self.starts_hash:
                self.logger.warn("starts hashes do not match")
        return False

    @property
    def savefile(self):
        '''
        Get path of the file, this DecompData object was last saved into or loaded from.
        returns None if object was never saved and not loaded or if attribute hashes in file do not match attribute hashes of object
        '''
        if (not self._savefile is None and pathlib.Path(self._savefile).is_file()):
            h5_file = h5py.File(self._savefile, "r")
            if self.check_hashes([ bytes.fromhex(h5_file[a].attrs["hash"]) for a in ["df","temps","spats","starts"] ]):
                return self._savefile
        return None

    @property
    def n_components(self):
        return self._spats.shape[0]

    @property
    def n_xaxis(self):
        return self._spats.shape[1]

    @property
    def n_yaxis(self):
        return self._spats.shape[2]

    @property
    def t_max(self):
        return self._temps.shape[0]

    @property
    def trials_n(self):
        ''' Returns number of trials'''
        return self._df.shape[0]

    @property
    def frame(self):
        return self._df

    #TODO both temporals only work for decompdata objects where phases have been applied to cut all trials into same length, otherwise reshaping doesnt work
    @property
    def temporals_z_scored(self):
        '''
        Get z-score of temporal components of DecompData object, reshaped into trials. Z-score is calculated using the mean and standart deviation of the full dataset, even after splitting the DecompData-object into conditions.
        '''
        try:
            return np.reshape(self._temps - self._mean * (1/self._stdev), (len(self), -1, self.n_components))
        except ValueError as err:
            if "cannot reshape array of size 0 into shape" in err.args[0]:
                return np.zeros((0, 0, self.n_components))
            else:
                raise

    @property
    def temporals(self):
        '''
        Get temporal components of DecompData object, reshaped into trials
        '''
        try:
            return np.reshape(self._temps, (len(self), -1, self.n_components))
        except ValueError as err:
            if "cannot reshape array of size 0 into shape" in err.args[0]:
                return np.zeros((0, 0, self.n_components))
            else:
                raise

    @property
    def temporals_flat(self):
        '''
        Get temporal components of DecompData object, kept as one timeseries
        '''
        return self._temps

    @property
    def spatials(self):
        return self._spats

    class PixelSlice:
        '''
        A helper class, designed to enable lazy slicing and accessing of recomposed pixel data by slicing spatials and temporals
        '''
        def __init__(self, temps, spats):
            self._temps = temps
            self._spats = spats

        def __getitem__(self, keys):
            if not isinstance(keys, tuple):
                keys = (keys,)
            temps = self._temps[keys[0], :]
            if len(keys) > 1:
                if np.array(keys[1]).ndim == 2:
                    spats = self._spats[:, keys[1]]
                else:
                    spats = self._spats[:, keys[1], keys[2] if len(keys) > 2 else slice(None, None, None)]
            else:
                spats = self._spats
            return np.tensordot(temps, spats, (-1, 0))

    def subsample(self,n,seed=None):
        '''
        Subsampling Trials to balance number of datapoints between different conditions
        :param n: Number of trials to sample
        :param seed: The seed for the rng
        '''
        rng = np.random.default_rng(seed)
        select_n = rng.choice(self.trials_n,size=n,replace=False)
        data = self[select_n]
        return data

    @property
    def pixel(self):
        '''
        Creates PixelSlice object, from which slices of recomposed pixel data can be accessed
        the first key is applied to the temporals, the second and third to the horizontal and vertical dimension of the spatials
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
            df = self._df.iloc[keys[0]]
        except NotImplementedError:
            df = self._df.loc[keys[0]]
        spats = self._spats
        try:
            # if keys[1] is bool us it as mask on aranged array to create array of frames to keep
            assert np.array(keys[1]).dtype == bool
            trial_frames = np.array(np.arange(len(keys[1]))[keys[1]])
            #self.logger.debug(f"frames {trial_frames}")
        except:
            try:
                # else use it to slice from aranged array
                #trial_frames = np.array(np.arange(np.max(np.diff(self._starts)))[keys[1]]) <- max can't work if we have some outliers in the frame length
                trial_frames = np.array(np.arange(np.min(np.diff(self._starts))+self._allowed_overlap)[keys[1]])
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
            data = DecompData(df, temps, spats, new_starts, spatial_labels=self._spat_labels, mean=self._mean, stdev=self._stdev, allowed_overlap=self._allowed_overlap)
        except AssertionError:
                self.logger.debug( starts.shape )
                self.logger.debug( selected_temps.shape )
                self.logger.debug( new_starts.shape )
                self.logger.debug( df )
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
        returns slice of self, containing all trials, where the trial data given by the keys of conditions match their corresponding values
        '''
        def check_attr(df, attr, val):
            if callable(val):
                return val(getattr( df, attr ))
            else:
                return getattr( df, attr ) == val

        select = True
        for attr, val in conditions.items():
            #self.logger.debug(f"dataframe columns {self._df.columns}")
            if isinstance(val,list):
                any = False
                for v in val:
                     any = any | check_attr(self._df, attr, v)
                select = select & any
            else:
                #self.logger.debug(f"{attr=} {val=} {self._df.loc[check_attr(self._df, attr, val)]}")
                select = select & check_attr(self._df, attr, val)
        #if(np.any(select)):
        return self[select]
        #else:
            #return None

    def dataset_from_sessions(self, sessions, dataset_id):
        '''
        Creates a DecompData object, only containing specified sessions and sets its datas_column to the specified dataset_id
        sessions should be a list of dicts, each containing a `subject_id` string and a `datetime` `numpy.datetime64` value
        '''
        subject_ids = [s['subject_id'] for s in sessions]
        dates = [s['datetime'] for s in sessions]
        def date_compare(d, date):
            return np.array(d, dtype=date.dtype) == date
        session_data = [self.get_conditional({'subject_id': subject_id}) for subject_id in subject_ids]
        session_data = [data.get_conditional({'date_time' : lambda d: date_compare(d,date) })
                                                    for date, data in zip(dates, session_data)]
        data = session_data[0]
        data.concat(session_data, overwrite=True)
        data._df[self.dataset_id_column] = dataset_id
        return session_data[0]

    def _op_data(self, a):
        df = self._df
        temps = self.temporals_flat
        spats = self.spatials
        starts = self._starts
        labels = self._spat_labels
        return df, temps, spats, starts, labels

    def __add__( a, b ):
        a_temps, b_temps, df, spats, starts, spat_labels = Data.binary_operation( a, b )
        return DecompData( df, a_temps+b_temps, spats, starts, spatial_labels=spat_labels, mean=self._mean, stdev=self._stdev, allowed_overlap=self._allowed_overlap)

    def __sub__( a, b ):
        a_temps, b_temps, df, spats, starts, spat_labels = Data.binary_operation( a, b )
        return DecompData( df, a_temps-b_temps, spats, starts, spatial_labels=spat_labels, mean=self._mean, stdev=self._stdev, allowed_overlap=self._allowed_overlap)

    def __mul__( a, b ):
        a_temps, b_temps, df, spats, starts, spat_labels = Data.binary_operation( a, b )
        return DecompData( df, a_temps*b_temps, spats, starts, spatial_labels=spat_labels, mean=self._mean, stdev=self._stdev, allowed_overlap=self._allowed_overlap)

class ConditionalData:
    def __init__( self, data, conditions ):
        if isinstance( conditions, dict ):
            self._data = { key : data.get_conditional(cond) for key, cond in conditions.items() }
        else:
            self._data = [ data.get_conditional(cond) for cond in conditions ]

    def keys(self):
        if isinstance( self._data, dict ):
            return self._data.keys()
        else:
            return range(len(self._data))

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys, slice(None, None, None))
        if isinstance( self._data, dict ):
            ret = self._data[keys[0]].__getitem__(keys[1:])
        else:
            dat = self._data[keys[0]]
            if isinstance(dat, Data ):
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
