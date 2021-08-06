import warnings
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import h5py


class Data(ABC):
    @abstractmethod
    def load(self, file, **kwargs):
        pass


# needs better name
class DecompData(Data):
    def __init__(self, df, temporal_comps, spatial_comps, trial_starts, cond_filter=None):

        self._df = df
        self._temps = temporal_comps
        self._spats = spatial_comps
        self._starts = trial_starts
        self._conditions = DecompData.Conditions(self, cond_filter)

    def save(self, file, temps_file=None, spats_file=None, df_label="df", temps_label="temps", spats_label="spats"):
        self._df.to_hdf(file, df_label, "w")
        h5_file = h5py.File(file, "a")
        if temps_file is None:
            temps = h5_file.create_dataset(temps_label, data=self._temps)
        else:
            with h5py.File(temps_file, "w") as h5_temps:
                temps = h5_temps.create_dataset(temps_label, data=self._temps)
            h5_file.attrs["temps_file"] = temps_file
        h5_file.attrs["temps_hash"] = self.temps_hash

        if spats_file is None:
            spats = h5_file.create_dataset(spats_label, data=self._spats)
        else:
            with h5py.File(spats_file, "w") as h5_spats:
                spats = h5_spats.create_dataset(spats_label, data=self._spats)
            h5_file.attrs["spats_file"] = spats_file
        h5_file.attrs["spats_hash"] = self.spats_hash

    def load(file, temps_file=None, spats_file=None, df_label="df", temps_label="temps", spats_label="spats"):
        df = pd.read_hdf(file, df_label)
        h5_file = h5py.File(file, "r")
        if temps_file is None:
            if temps_label in h5_file:
                temps = np.array(h5_file[temps_label])
            elif "temps_file" in h5_file.attrs:
                with h5py.File(h5_file.attrs["temps_file"], "r") as h5_temps:
                    temps = np.array(h5_temps[temps_label])
                if "temps_hash" in h5_file.attrs and h5_file.attrs["temps_hash"] != hash(temps.data.tobytes()):
                    warnings.warn("Component hashes do not match", Warning)
            else:
                raise ValueError
        else:
            with h5py.File(temps_file, "r") as h5_temps:
                temps = np.array(h5_temps[temps_label])
            if "temps_hash" in h5_file.attrs and h5_file.attrs["temps_hash"] != hash(temps.data.tobytes()):
                warnings.warn("Component hashes do not match", Warning)

        if spats_file is None:
            if spats_label in h5_file:
                spats = np.array(h5_file[spats_label])
            elif "spats_file" in h5_file.attrs:
                with h5py.File(h5_file.attrs["spats_file"], "r") as h5_spats:
                    spats = np.array(h5_spats[spats_label])
                if "spats_hash" in h5_file.attrs and h5_file.attrs["spats_hash"] != hash(spats.data.tobytes()):
                    warnings.warn("Feature hashes do not match", Warning)
            else:
                raise ValueError
        else:
            with h5py.File(spats_file, "r") as h5_spats:
                spats = np.array(h5_spats[spats_label])
            if "spats_hash" in h5_file.attrs and h5_file.attrs["spats_hash"] != hash(spats.data.tobytes()):
                warnings.warn("Feature hashes do not match", Warning)
        return DecompData(df, temps, spats)

    @property
    def temps_hash(self):
        return hash(self._temps.data.tobytes())

    @property
    def spats_hash(self):
        return hash(self._spats.data.tobytes())

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
        return self._temps.reshape(len(self), -1, self.n_components)

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
            df = self._df
        else:
            if not len(keys) > 1:
                keys[1] = keys[1:]
            df = self._df[keys[1]]
        spats = self._spats
        try:
            assert np.array(keys[0]).dtype == bool
            trial_frames = np.array(np.arange(len(keys[0]))[keys[0]])
        except:
            trial_frames = np.array(np.arange(np.max(np.diff(self._starts)))[keys[0]])
        starts = np.array(self._starts[:-1][keys[1]])
        selected_temps = np.array(trial_frames[np.newaxis, :] + starts[:, np.newaxis], dtype=int)
        new_starts = np.insert(np.cumsum(np.diff(selected_temps[:, (0, -1)]) + 1), 0, 0)
        temps = self._temps[selected_temps.flatten()]
        return DecompData(df, temps, spats, new_starts)

    def __getattr__(self, key):
        # if key in self._df.keys():
        return getattr(self._df, key)

    # else:
    # raise AttributeError

    def __len__(self):
        return self._df.__len__()


    '''
    @property
    def conditions(self):
        return [self[:,filter] for filter in self._cond_filters]

    def setConditions(self, cond_filters):
        self._cond_filters = cond_filters

    #def setCondition(self, ):
    '''
    #
    class Conditions:
        def __init__(self, parent, cond_filters):
            if cond_filters is None:
                cond_filters = []
            self._cond_filters = cond_filters

        def __setitem__(self, parent, cond_filters):
            print("GetMethod of Conditions")
            if cond_filters is None:
                cond_filters = []
            self._cond_filters = cond_filters

        def __getitem__(self, parent):
            print("GetMethod of Conditions")
            return [parent[f] for f in self._cond_filters]

        def __len__(self):
            return len(self._cond_filters)

    #Returns filtered conditions
    @property
    def conditions(self):
        #return self._conditions
        return [self[:,f] for f in self._conditions]

    @conditions.setter
    def conditions(self,cond_filters):
        self._conditions = cond_filters

    #Returns conditions filter
    @property
    def condition_filter(self):
        return self._conditions._cond_filters

    @condition_filter.setter #just another interface
    def condition_filter(self,cond_filters):
        self._conditions = self._cond_filters

