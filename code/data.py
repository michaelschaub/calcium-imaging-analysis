import warnings
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import h5py

class Data(ABC):
    @abstractmethod
    def load( self, file, **kwargs ):
        pass


# needs better name
class SVDData(Data):
    def __init__( self, frame, components, features ):
        self._frame = frame
        self._comps = components
        self._feats = features

    def save( self, file, comp_file=None, feat_file=None, frame_label="frame", comp_label="comps", feat_label="feats" ):
        self._frame.to_hdf( file, frame_label, "w" )
        h5_file = h5py.File( file, "a" )
        if comp_file is None:
            comps = h5_file.create_dataset(comp_label, data=self._comps)
        else:
            with h5py.File( comp_file, "w" ) as h5_comps:
                comps = h5_comps.create_dataset(comp_label, data=self._comps)
            h5_file.attrs["comp_file"] = comp_file
        h5_file.attrs["comp_hash"] = self.comp_hash

        if feat_file is None:
            feats = h5_file.create_dataset(feat_label, data=self._feats)
        else:
            with h5py.File( feat_file, "w" ) as h5_feats:
                feats = h5_feats.create_dataset(feat_label, data=self._feats)
            h5_file.attrs["feat_file"] = feat_file
        h5_file.attrs["feat_hash"] = self.feat_hash

    def load( file, comp_file=None, feat_file=None, frame_label="frame", comp_label="comps", feat_label="feats" ):
        frame = pd.read_hdf( file, frame_label )
        h5_file = h5py.File( file, "r")
        if comp_file is None:
            if comp_label in h5_file:
                comps = np.array(h5_file[comp_label])
            elif "comp_file" in h5_file.attrs:
                with h5py.File( h5_file.attrs["comp_file"], "r" ) as h5_comps:
                    comps = np.array(h5_comps[comp_label])
                if "comp_hash" in h5_file.attrs and h5_file.attrs["comp_hash"] != hash(comps.data.tobytes()):
                    warnings.warn( "Component hashes do not match", Warning )
            else:
                raise ValueError
        else:
            with h5py.File( comp_file, "r" ) as h5_comps:
                comps = np.array(h5_comps[comp_label])
            if "comp_hash" in h5_file.attrs and h5_file.attrs["comp_hash"] != hash(comps.data.tobytes()):
                warnings.warn( "Component hashes do not match", Warning )

        if feat_file is None:
            if feat_label in h5_file:
                feats = np.array(h5_file[feat_label])
            elif "feat_file" in h5_file.attrs:
                with h5py.File( h5_file.attrs["feat_file"], "r" ) as h5_feats:
                    feats = np.array(h5_feats[feat_label])
                if "feat_hash" in h5_file.attrs and h5_file.attrs["feat_hash"] != hash(feats.data.tobytes()):
                    warnings.warn( "Feature hashes do not match", Warning )
            else:
                raise ValueError
        else:
            with h5py.File( feat_file, "r" ) as h5_feats:
                feats = np.array(h5_feats[feat_label])
            if "feat_hash" in h5_file.attrs and h5_file.attrs["feat_hash"] != hash(feats.data.tobytes()):
                warnings.warn( "Feature hashes do not match", Warning )
        return SVDData( frame, comps, feats )

    @property
    def comps( self ):
        return self._comps

    @comps.setter
    def comps( self, comps ):
        self._comps = comps

    @property
    def feats( self ):
        return self._feats

    @feats.setter
    def feats( self, feats ):
        self._feats = feats

    @property
    def comp_hash( self ):
        return hash(self._comps.data.tobytes())

    @property
    def feat_hash( self ):
        return hash(self._feats.data.tobytes())

    @property
    def n_components( self ):
        return self._feats.shape[0]

    @property
    def n_xaxis( self ):
        return self._feats.shape[1]

    @property
    def n_yaxis( self ):
        return self._feats.shape[2]

    @property
    def t_max( self ):
        return self._comps.shape[0]

    class PixelSlice:
        def __init__(self, comps, feats ):
            self._comps = comps
            self._feats = feats

        def __getitem__( self, keys ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            comps = self._comps[ keys[0], :]
            if len(keys)>1:
                if np.array(keys[1]).ndim == 2:
                    feats = self._feats[ :, keys[1] ]
                else:
                    feats = self._feats[ :, keys[1] ,keys[2] if len(keys)>2 else slice(None,None,None)]
            else:
                feats = self._feats
            return np.tensordot( comps, feats, (-1,0) )

    @property
    def pixel( self ):
        return SVDData.PixelSlice( self._comps, self._feats )


    def __getattr__( self, key ):
        #if key in self._frame.keys():
            return getattr(self._frame, key)
        #else:
            #raise AttributeError
