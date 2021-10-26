import warnings
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import h5py

#For BrainAlignment
import scipy.io, scipy.ndimage

import sys
from pathlib import Path
#sys.path.append(Path(__file__).parent)


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


class DecompData(Data):
    def __init__(self, df, temporal_comps, spatial_comps, trial_starts, cond_filter=None, trans_params=None):
        assert len(df) != trial_starts.shape[0]-1, (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})\n\t(maybe remove last entry?)")
        assert len(df) == trial_starts.shape[0], (
            f"DataFrame df and trial_starts do not have matching length ({len(df)} != {len(trial_starts)})")
        self._df = df
        self._temps = temporal_comps
        self._spats = spatial_comps if trans_params is None else self.align_spatials(spatial_comps, trans_params)
        self._starts = trial_starts

        if cond_filter is None:
            cond_filter = []
        self._cond_filters = cond_filter  # DecompData.Conditions(self, cond_filter)

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

    #Auslagern
    def align_spatials(self, spatials, trans_params):
        f , h , w = spatials.shape #org shape

        #Attend bitmap as last frame
        spatials = np.append(spatials,np.ones((1,h,w)),axis=0)

        #Offset instead of Nans as interpolation is used
        min = np.nanmin(spatials)

        eps = 0 #np.finfo(np.float32).eps
        #offset = 2*eps # 10
        #spatials = spatials - min #+ offset
        print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))
        #Rotation
        print("Rotation")
        spatials = scipy.ndimage.rotate(spatials,trans_params['angleD'], axes=(2,1), reshape=True, cval= -eps)
        print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))

        ### introduces weird aliasing along edges due to interpolation
        #Scale
        print("Scale/Zoom")
        spatials = scipy.ndimage.zoom(spatials, (1,trans_params['scaleConst'],trans_params['scaleConst']),order=1,cval= -eps) #slow
        print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))

        #Translate
        print("Translate/Shift")
        spatials = scipy.ndimage.shift(spatials, np.insert(np.flip(trans_params['tC']),0,0),cval= -eps, order=1, mode='constant') #slow
        ### ---
        print("Min/Max Value:",np.nanmin(spatials),np.nanmax(spatials))

        #Remove offset

        bitmask = spatials[-1,:,:]<0.5 #set bitmap as all elements that were interpolated under 0.5
        spatials = np.delete(spatials,-1,axis=0) #delete Bitmap from spatials

        bitmask = np.broadcast_to(bitmask,spatials.shape) #for easier broadcasting, is not in memory
        np.putmask(spatials,bitmask,np.NAN) #set all elements of bitmap to NAN


        #Crop
        print("Crop")
        n_spatials , h_new , w_new = spatials.shape
        trim_h = int(np.floor((h_new - h) / 2 ))
        trim_w = int(np.floor((w_new - w) / 2 ))

        #Eleganter lösen, hier nur 1 zu 1 matlab nachgestellt
        if trans_params['scaleConst'] < 1:
            if trim_h < 0:
                temp_spats = np.full((n_spatials, h, w_new),np.NAN)
                temp_spats[:,abs(trim_h):abs(trim_h)+h_new, :] = spatials
                spatials = temp_spats
            else:
                spatials = spatials[:,trim_h:trim_h + h, :]

            n_spatials , h_new , w_new = spatials.shape
            if trim_w < 0:
                temp_spats = np.full((n_spatials, h_new, w),np.NAN)
                temp_spats[:,:,abs(trim_w):abs(trim_w) + w_new] = spatials
                spatials = temp_spats
            else:
                spatials = spatials[:,:,trim_w:trim_w+w]

        else:
            spatials = spatials[:,trim_h:trim_h + h, trim_w:trim_w+w]

        return spatials

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
        try:
            return self._temps.reshape(len(self), -1, self.n_components)
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
        # if key in self._df.keys():
        return getattr(self._df, key)

        # else:
        # raise AttributeError

    def __len__(self):
        return self._df.__len__()

    # Returns filtered conditions
    @property
    def conditions(self):
        return self._conditional

    @conditions.setter
    def conditions(self, conditions):
        self._cond_filters = conditions
        self._conditional = ConditionalData( self, conditions )

    # Returns conditions filter
    @property
    def condition_filters(self):
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
