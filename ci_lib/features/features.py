import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

from ci_lib import Data, DecompData
from ci_lib.loading import reproducable_hash, load_h5, save_h5

import pathlib
from enum import Enum


class Feature_Type(Enum):
    NODE = 0
    UNDIRECTED = 1
    DIRECTED = 2
    TIMESERIES = 3

class Features:
    def __init__(self, data, feature, file=None):
        self.data = data
        self._feature = feature
        self._savefile = file

    def flatten(self):
        '''flatten contained feauture to one trial and one feature dimension'''
        pass

    def expand(self, data=None):
        '''expand feature into same shape as temporals in Data (for computation)'''
        if data is None:
            data = self.data
        feats = self.feature
        feat = np.empty((data.temporals_flat.shape[0], *feats.shape[1:]), dtype=float)
        starts = list(data._starts)
        starts.append(-1)
        for i in range(len(starts) - 1):
            feat[starts[i]:starts[i + 1]] = feats[i]
        return feat

    @property
    def feature(self):
        return self._feature

    @property
    def mean(self):
        '''
        Create FeatureMean object taking trial means over selfs features
        '''
        return FeatureMean.create(self)

    @property
    def hash(self):
        '''
        reproducable hashsum of features to compare with others
        WARNING: MAY NOT WORK, STILL IN DEVELOPMENT
        '''
        return reproducable_hash(self._feature)

    @property
    def data(self):
        '''
        data object from with this feature is computated,
        is retrieved from Data.LOADED_DATA by hash or loaded from file, when not set directly
        '''
        if not hasattr(self, '_data'):
            if hasattr(self, '_data_hash') and self._data_hash in Data.LOADED_DATA:
                self._data = Data.LOADED_DATA[self._data_hash]
            elif hasattr(self, '_data_file'):
                self._data = DecompData.load(self._data_file)
            else:
                raise ValueError("Data object of this Feature could not be reconstructed")
        return self._data

    @data.setter
    def data(self, data):
        '''
        sets data object, as well as hash and savefile metadata
        data should either be data object, integer data hash or pathlike
        '''
        if hasattr(self, '_data'):
            raise AttributeError("Feature already has data object set.")
        if isinstance( data, Data ):
            self._data = data
            self._data_hash = data.hash.digest()
            self._data_file = data.savefile
        elif type(data) == bytes:
            self.data_hash = data
        else:
            try:
                self.data_file = str(pathlib.Path(data))
            except TypeError:
                raise AttributeError(f"Feature data cannot be set via {type(data)}.") from None

    @property
    def data_hash(self):
        if not hasattr(self, '_data'):
            return self._data_hash
        else:
            return self._data.hash.digest()

    @data_hash.setter
    def data_hash(self, data_hash):
        if not hasattr(self, '_data'):
            self._data_hash = data_hash
        elif  self._data.hash.digest() != data_hash:
            raise AttributeError("Feature already has data object set.")

    @property
    def data_file(self):
        if not hasattr(self, '_data'):
            return self._data_file
        else:
            return self._data.savefile

    @data_file.setter
    def data_file(self, data_file):
        if not hasattr(self, '_data') and not hasattr(self, '_data_hash'):
            self._data_file = data_file
        elif not ( hasattr(self, '_data') and str(pathlib.Path(self._data.file)) == str(pathlib.Path(data_file))):
            raise AttributeError("Feature already has data object or hash set.")

    def save(self, file, data_file=None):
        '''
        '''
        h5_file = save_h5( self, file, { "feature": self._feature } )
        h5_file.attrs["data_hash"] = self.data_hash.hex()
        if self.data.savefile is None:
            if data_file is None:
                path = pathlib.Path(file)
                data_file = path.parent / f"data.{path.stem}{path.suffix}"
            self.data.save(data_file)
        assert (self._data.savefile is not None), "Failure in saving underlaying data object!"
        h5_file.attrs["data_file"] = str(self._data.savefile)
        self._savefile = file

    @classmethod
    def load(Class, file, data_file=None, feature_hash=None, try_loaded=False):
        if try_loaded and feature_hash is not None and feature_hash in Features.LOADED_FEATURES:
            feat = Features.LOADED_FEATURES[feature_hash]
        else:
            h5_file, feature = load_h5( file, labels=["feature"] )
            data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            if try_loaded and data_hash in Data.LOADED_DATA:
                data = Data.LOADED_DATA[data_hash]
            elif data_file is None:
                data = h5_file.attrs["data_file"]
            feat = Class(data, feature, file)
            feat.data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            Features.LOADED_FEATURES[feat.hash.digest()] = feat
        return feat

    LOADED_FEATURES = {}

    @property
    def type(self):
        return self._type


class Raws(Features):
    _type = Feature_Type.TIMESERIES

    def create(data, max_comps=None, logger=LOGGER):
        feat = Raws(data, data.temporals[:, :, :max_comps])
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.reshape(feat, (feat.shape[0], -1))

    @property
    def pixel(self):
        return DecompData.PixelSlice(np.reshape(self._feature, (-1, *self._feature[2:])),
                                     self.data._spats[:self._feature.shape[2]])

    @property
    def ncomponents(self):
        return self._feature.shape[-1]


class FeatureMean(Features):
    def create(base, logger=LOGGER):
        feature = np.mean(base.feature, axis=0)[None,:].reshape((1, *base.feature.shape[1:]))
        feat = FeatureMean(base.data, feature )
        feat._base_feature = base
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return self._base_feature.flatten(self._feature)

    @property
    def pixel(self):
        if isinstance(self._base_feature, Raws):
            return DecompData.PixelSlice(np.reshape(self._feature, (self._feature.shape[1:])),
                                         self._data._spats[:self._feature.shape[2]])
        elif isinstance(self._base_feature, Means):
            return DecompData.PixelSlice(self._feature,
                                         self._data._spats[:self._feature.shape[1]])
        else:
            raise AttributeError

    def save(self, file, data_file=None):
        '''
        not yet implemented
        '''
        pass

    @classmethod
    def load(Class, file, data_file=None, feature_hash=None, try_loaded=False):
        '''
        not yet implemented
        '''
        pass

    @property
    def ncomponents(self):
        return self._feature.shape[-1]
