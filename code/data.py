import warnings
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import h5py
import pathlib

import sys

from loading import reproducable_hash, load_h5, save_h5

class Data(ABC):
    @abstractmethod
    def load(self, file, **kwargs):
        pass

    def binary_operation( a, b ):
        notNone = lambda x,y : y if x is None else x
        try:
            a_df, a_temps, a_spats, a_starts = a._op_data(b)
        except AttributeError:
            a_df = None
            a_temps = a
            a_spats = None
            a_starts = None
        try:
            b_df, b_temps, b_spats, b_starts = b._op_data(a)
        except AttributeError:
            b_df = None
            b_temps = b
            b_spats = None
            b_starts = None
        return a_temps, b_temps, notNone(a_df,b_df), notNone(a_spats, b_spats), notNone(a_starts, b_starts)

    LOADED_DATA = {}


# needs better name
class DecompData(Data):
    def __init__(self, df, temporal_comps, spatial_comps, trial_starts, cond_filter=None, savefile=None, read_only=True):
        assert len(df) != trial_starts.shape[0]-1, (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})\n\t(maybe remove last entry?)")
        assert len(df) == trial_starts.shape[0], (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})")
        self._df = df
        self._temps = temporal_comps
        self._spats = spatial_comps
        self._starts = trial_starts

        if cond_filter is None:
            cond_filter = []
        self.conditions = cond_filter  # DecompData.Conditions(self, cond_filter)
        self._savefile = savefile

        if read_only:
            self._temps.flags.writeable = False
            self._spats.flags.writeable = False
            self._starts.flags.writeable = False


    def save(self, file, temps_file=None, spats_file=None, starts_file=None, temps_label="temps", spats_label="spats", starts_label="starts" ):
        h5_file = save_h5( self, file, df=self._df,
                            attributes=[self._temps, self._spats, self._starts],
                            attr_files=[temps_file, spats_file, starts_file ],
                            labels=[temps_label, spats_label, starts_label ],
                            hashes=[self.df_hash, self.temps_hash, self.spats_hash, self.starts_hash ] )
        self._savefile = file

    @classmethod
    def load(Class, file, temps_file=None, spats_file=None, starts_file=None, df_label="df", temps_label="temps", spats_label="spats", starts_label="starts", data_hash=None, try_loaded=False):
        if try_loaded and data_hash is not None and data_hash in Data.LOADED_DATA:
            data = Data.LOADED_DATA[h5_file.attrs["data_hash"]]
        else:
            _, df, temps, spats, starts = load_h5( file,
                                attr_files=[temps_file, spats_file, starts_file ],
                                labels=[temps_label, spats_label, starts_label ])
            data = Class(df, temps, spats, starts, savefile=file)
            Data.LOADED_DATA[data.hash] = data
        return data

    @property
    def hash(self):
        return hash( (self.df_hash, self.temps_hash, self.spats_hash, self.starts_hash) )

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

    def check_hashes(self, hashes, warn=True):
        if hash(tuple(hashes)) == self.hash:
            return True
        elif warn:
            if hashes[0] is not None and hashes[0] != self.df_hash:
                warnings.warn("df hashes do not match", Warning)
            if hashes[1] is not None and hashes[1] != self.temps_hash:
                warnings.warn("temps hashes do not match", Warning)
            if hashes[2] is not None and hashes[2] != self.spats_hash:
                warnings.warn("spats hashes do not match", Warning)
            if hashes[3] is not None and hashes[3] != self.starts_hash:
                warnings.warn("starts hashes do not match", Warning)
        return False

    @property
    def savefile(self):
        if (not self._savefile is None and pathlib.Path(self._savefile).is_file()):
            h5_file = h5py.File(self._savefile, "r")
            if self.check_hashes([ h5_file.attrs[f"{a}_hash"] for a in ["df","temps","spats","starts"] ]):#, warn=False):
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
        if not isinstance(keys, tuple):
            keys = (keys, slice(None, None, None))
        elif len(keys) < 2:
            keys = (keys[0], slice(None,None,None))
        df = self._df[keys[0]]
        spats = self._spats
        try:
            assert np.array(keys[1]).dtype == bool
            trial_frames = np.array(np.arange(len(keys[1]))[keys[1]])
        except:
            try:
                trial_frames = np.array(np.arange(np.max(np.diff(self._starts)))[keys[1]])
            except ValueError as err:
                if "zero-size array to reduction operation maximum" in err.args[0]:
                    print("Warning: Data has size zero")
                    trial_frames = np.array([])
                else:
                    raise
        # starts of selected frames in old temps
        starts = np.array(self._starts[keys[0]])

        # indices of temps in all selected frames (2d)
        selected_temps = np.array(trial_frames[np.newaxis, :] + starts[:, np.newaxis], dtype=int)

        # starts of selected frames in new temps
        new_starts = np.insert(np.cumsum(np.diff(selected_temps[:-1, (0, -1)]) + 1), 0, 0)

        temps = self._temps[selected_temps.flatten()]
        try:
            data = DecompData(df, temps, spats, new_starts)
        except AssertionError:
                print( starts.shape )
                print( selected_temps.shape )
                print( new_starts.shape )
                print( df )
                raise
        return data

    def __getattr__(self, key):
        try:
            return getattr(self._df, key)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from None

    def __len__(self):
        return self._df.__len__()

    @property
    def conditions(self):
        '''Returns filtered conditions'''
        return self._conditional

    @conditions.setter
    def conditions(self, conditions):
        self._cond_filters = conditions
        self._conditional = ConditionalData( self, conditions )

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
            select = select & (getattr( self._df, attr ) == val)
        return self[select]

    def _op_data(self, a):
        df = self._df
        temps = self.temporals_flat
        spats = self.spatials
        starts = self._starts
        return df, temps, spats, starts

    def __add__( a, b ):
        a_temps, b_temps, df, spats, starts = Data.binary_operation( a, b )
        return DecompData( df, a_temps+b_temps, spats, starts )

    def __sub__( a, b ):
        a_temps, b_temps, df, spats, starts = Data.binary_operation( a, b )
        return DecompData( df, a_temps-b_temps, spats, starts )

    def __mul__( a, b ):
        a_temps, b_temps, df, spats, starts = Data.binary_operation( a, b )
        return DecompData( df, a_temps*b_temps, spats, starts )

class ConditionalData:
    def __init__( self, data, conditions ):
        if isinstance( conditions, dict ):
            self._data = { key : data.get_conditional(cond) for key, cond in conditions.items() }
        else:
            self._data = [ data.get_conditional(cond) for cond in conditions ]

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
