from abc import ABC, abstractmethod, abstractproperty
import numpy as np
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

    LOADED_DATA = {}


class DecompData(Data):
    def __init__(self, df, temporal_comps, spatial_comps, trial_starts, cond_filter=None, trans_params=None, savefile=None, read_only=True, spatial_labels=None, logger=None):
        self.logger = LOGGER if logger is None else logger
        assert len(df) != trial_starts.shape[0]-1, (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})\n\t(maybe remove last entry?)")
        assert len(df) == trial_starts.shape[0], (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})")
        self._df = df
        self._temps = temporal_comps
        self._spats = spatial_comps if trans_params is None else align_spatials(spatial_comps,trans_params, logger=self.logger)
        self._starts = trial_starts
        self._spat_labels = spatial_labels


        if cond_filter is None:
            cond_filter = []
        self.conditions = cond_filter  # DecompData.Conditions(self, cond_filter)
        self._savefile = savefile

        if read_only:
            self._temps.flags.writeable = False
            self._spats.flags.writeable = False
            self._starts.flags.writeable = False


    #Used for parcellations
    def update(self,temporal_comps, spatial_comps, spatial_labels=None):
        self._temps = temporal_comps
        self._spats = spatial_comps
        self._spat_labels = spatial_labels

    def save(self, file ):
        h5_file = save_h5( self, file, {"df"    : self._df,
                                        "temps" : self._temps,
                                        "spats" : self._spats,
                                        "starts" : self._starts,
                                        "labels":self._spat_labels}, logger=self.logger)
        self._savefile = file

    @classmethod
    def load(Class, file, data_hash=None, try_loaded=False, logger=LOGGER):
        if try_loaded and data_hash is not None and data_hash in Data.LOADED_DATA:
            data = Data.LOADED_DATA[data_hash]
        else:
            _, df, temps, spats, starts, spat_labels = load_h5( file, labels=["df", "temps", "spats", "starts","labels"], logger=logger)
            data = Class(df, temps, spats, starts, spatial_labels=spat_labels, savefile=file, logger=logger)
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
    def temporals(self):
        try:
            return np.reshape(self._temps, (len(self), -1, self.n_components))
        except ValueError as err:
            if "cannot reshape array of size 0 into shape" in err.args[0]:
                return np.zeros((0, 0, self.n_components))
            else:
                raise

    @property
    def temporals_flat(self):
        return self._temps

    @property
    def spatials(self):
        return self._spats

    class PixelSlice:
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

    @property
    def pixel(self):
        return DecompData.PixelSlice(self._temps, self._spats)

    def __getitem__(self, keys):
        print("keys")
        print(keys)
        if not isinstance(keys, tuple):
            keys = (keys, slice(None, None, None))
        elif len(keys) < 2:
            keys = (keys[0], slice(None,None,None))
        df = self._df[keys[0]]
        spats = self._spats
        try:
            assert np.array(keys[1]).dtype == bool
            trial_frames = np.array(np.arange(len(keys[1]))[keys[1]])
            print("frames")
            print(trial_frames)
        except:
            try:
                print("self starts")
                print(self._starts)
                print(np.diff(self._starts))
                print("max",np.max(np.diff(self._starts)))
                print("min",np.min(np.diff(self._starts)))

                #trial_frames = np.array(np.arange(np.max(np.diff(self._starts)))[keys[1]]) <- max can't work if we have some outliers in the frame length
                trial_frames = np.array(np.arange(np.min(np.diff(self._starts)))[keys[1]])

                print("frames")
                print(trial_frames)
            except ValueError as err:
                if "zero-size array to reduction operation maximum" in err.args[0]:
                    self.logger.warn("Data has size zero")
                    trial_frames = np.array([])
                else:
                    raise
        # starts of selected frames in old temps
        starts = np.array(self._starts[keys[0]])
        print("starts")
        print(starts)

        # indices of temps in all selected frames (2d)
        selected_temps = np.array(trial_frames[np.newaxis, :] + starts[:, np.newaxis], dtype=int)
        print("frames + starts")
        print(selected_temps)

        # starts of selected frames in new temps
        new_starts = np.insert(np.cumsum(np.diff(selected_temps[:-1, (0, -1)]) + 1), 0, 0)

        print(self._temps.shape)
        temps = self._temps[selected_temps.flatten()]
        try:
            data = DecompData(df, temps, spats, new_starts, spatial_labels=self._spat_labels)
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
        select = True
        for attr, val in conditions.items():
            if isinstance(val,list):
                any = False
                for v in val:
                     any = any | (getattr( self._df, attr ) == v)
                select = select & any
            else:
                select = select & (getattr( self._df, attr ) == val)
        return self[select]

    def _op_data(self, a):
        df = self._df
        temps = self.temporals_flat
        spats = self.spatials
        starts = self._starts
        labels = self._spat_labels
        return df, temps, spats, starts, labels

    def __add__( a, b ):
        a_temps, b_temps, df, spats, starts, spat_labels = Data.binary_operation( a, b )
        return DecompData( df, a_temps+b_temps, spats, starts, spatial_labels=spat_labels)

    def __sub__( a, b ):
        a_temps, b_temps, df, spats, starts, spat_labels = Data.binary_operation( a, b )
        return DecompData( df, a_temps-b_temps, spats, starts, spatial_labels=spat_labels)

    def __mul__( a, b ):
        a_temps, b_temps, df, spats, starts, spat_labels = Data.binary_operation( a, b )
        return DecompData( df, a_temps*b_temps, spats, starts, spatial_labels=spat_labels)

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