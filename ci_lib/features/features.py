'''
This module contains the some base classes for handling features calculated from DecompData objects.
'''

import logging
import pathlib
from enum import Enum
import numpy as np
import pandas as pd

from ci_lib.data import DecompData, LOADED_DATA
from ci_lib.loading import reproducable_hash, load_h5, save_h5

LOGGER = logging.getLogger(__name__)


class FeatureType(Enum):
    '''
    An enum describing the structure and possible representation of a Feature object
    '''
    NODE = 0
    UNDIRECTED = 1
    DIRECTED = 2
    TIMESERIES = 3
    def __eq__(self, other):
        return self.value == other.value

class Features:
    '''
    A class containing a feature calculated from a DecompData object
    Used as a base class for specific features, like temporal mean, covariance, etc.
    '''

    _type : FeatureType

    def __init__(self, frame, data, feature, file=None, time_resolved=False, full = False):
        assert len(frame) == feature.shape[0], f"DataFrame frame and feature do not have matching length \
                    ({len(frame)} != {feature.shape[0]})"
        self.data = data
        self._df = frame.reset_index(drop=True)
        self._feature = feature
        self._savefile = file

        self._time_resolved = time_resolved
        self._full = full




    def flatten(self):
        '''flatten contained feature to one trial and one feature dimension'''


    def concat(self, features, overwrite=False):
        ''' concats feature values from List of Features to this feature'''
        if not isinstance(features, list):
            features = [features]
        if not overwrite:
            features = [self, *features]
        self._feature = np.concatenate([f.feature for f in features],axis=0)
        self._df = pd.concat([f._df for f in features]).reset_index(drop=True)

        LOGGER.debug("Concated feature.shape=%s", self.feature.shape)
        LOGGER.debug("Concated   frame.shape=%s", self.frame.shape)
        return type(self)(self._df, self.data, self._feature)

    def expand(self, data=None):
        '''expand feature into same shape as temporals in Data (for computation)'''
        if data is None:
            data = self.data
        feats = self.feature
        feat = np.empty((data.temporals_flat.shape[0], *feats.shape[1:]), dtype=float)
        starts = list(data.trial_starts)
        starts.append(-1)
        for i in range(len(starts) - 1):
            feat[starts[i]:starts[i + 1]] = feats[i]
        return feat

    @property
    def frame(self):
        '''The pandas DataFrame containing the trial data'''
        return self._df

    @frame.setter
    def frame(self,frame):
        self._df = frame.reset_index(drop=True)

    @property
    def feature(self):
        '''The contained feature data, should be a numpy array'''
        return self._feature

    @feature.setter
    def feature(self,value):
        self._feature = value

    @property
    def trials_n(self):
        ''' The number of trials, the first dimension of the feature array'''
        return self._feature.shape[0]

    @property
    def timepoints(self):
        '''
        The number of frames, the second dimension of the feature array,
        if it is time resolved and not full (decode timepoints together), otherwise None
        '''
        if self._time_resolved and not self._full:
            return self._feature.shape[1]
        return None

    @property
    def is_time_resolved(self):
        #TODO fill this out
        '''Not sure why we have this'''
        return self._time_resolved

    def __getitem__(self, key):
        try:
            data_frame = self._df.iloc[key]
        except NotImplementedError:
            data_frame = self._df.loc[key]
        self._df = data_frame
        self._feature = self._feature[key]
        #TODO make copy before
        return self

    def subsample(self,size,seed=None):
        '''
        Subsampling Trials to balance number of datapoints between different conditions
        :param size: Number of trials to sample
        :param seed: The seed for the rng
        '''
        rng = np.random.default_rng(seed)
        select_n = rng.choice(self.trials_n,size=size,replace=False)
        return self[select_n]

    def get_conditional(self, conditions):
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
                select = select & check_attr(self.frame, attr, cond_val)
        #if(np.any_matching(select)):
        return self[select]

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
        is retrieved from LOADED_DATA by hash or loaded from file, when not set directly
        '''
        if not hasattr(self, '_data'):
            if hasattr(self, '_data_hash') and self._data_hash in LOADED_DATA:
                self._data = LOADED_DATA[self._data_hash]
            elif hasattr(self, '_data_file'):
                self._data = DecompData(pd.DataFrame(),np.array([]),np.array([]),np.array([]))#DecompData.load(self._data_file)
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
        if isinstance( data, DecompData ):
            self._data = data
            self._data_hash = data.hash.digest()
            self._data_file = data.savefile
        elif isinstance(data, bytes):
            self.data_hash = data
        else:
            try:
                self.data_file = str(pathlib.Path(data))
            except TypeError:
                raise AttributeError(f"Feature data cannot be set via {type(data)}.") from None

    @property
    def data_hash(self):
        '''The hash of the underlying DecompData object'''
        if not hasattr(self, '_data'):
            return self._data_hash
        return self._data.hash.digest()

    @data_hash.setter
    def data_hash(self, data_hash):
        if not hasattr(self, '_data'):
            self._data_hash = data_hash
        elif  self._data.hash.digest() != data_hash:
            raise AttributeError("Feature already has data object set.")

    @property
    def data_file(self):
        '''The file where the underlying DecompData object is saved, if any'''
        if not hasattr(self, '_data'):
            return self._data_file
        return self._data.savefile

    @data_file.setter
    def data_file(self, data_file):
        if not hasattr(self, '_data') and not hasattr(self, '_data_hash'):
            self._data_file = data_file
        elif ( not ( hasattr(self, '_data')
                and str(pathlib.Path(self._data.file)) == str(pathlib.Path(data_file)))):
            raise AttributeError("Feature already has data object or hash set.")

    def save(self, file, data_file=None):
        '''
        Save this Feature object to file `file` as an h5
        '''
        h5_file = save_h5( self, file, { "df": self._df, "feature": self._feature,
                                        "time_resolved":np.asarray(self._time_resolved) } )
        h5_file.attrs["data_hash"] = self.data_hash.hex()
        
        if self.data.savefile is None:
            if data_file is None:
                path = pathlib.Path(file)
                data_file = path.parent / f"data.{path.stem}{path.suffix}"
            #self.data.save(data_file)
        #assert (self._data.savefile is not None), "Failure in saving underlaying data object!"
        #self._data.savefile = data_file  
        h5_file.attrs["data_file"] = str(data_file) #self._data.savefile)
        
        self._savefile = file

    @classmethod
    def load(cls, file, data_file=None, feature_hash=None, try_loaded=False):
        '''
        Loads a saved Feature object from file `file`
        '''
        if try_loaded and feature_hash is not None and feature_hash in Features.LOADED_FEATURES:
            feat = Features.LOADED_FEATURES[feature_hash]
        else:
            loaded = load_h5( file, labels=["df", "feature","time_resolved"] )
            # pylint: disable-next=unbalanced-tuple-unpacking
            h5_file, frame, feature, time_resolved = loaded
            data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            if try_loaded and data_hash in LOADED_DATA:
                data = LOADED_DATA[data_hash]
            elif data_file is None:
                data = h5_file.attrs["data_file"]
            feat = cls(frame, data, feature, file, time_resolved=time_resolved)
            feat.data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            Features.LOADED_FEATURES[feat.hash.digest()] = feat
        return feat

    LOADED_FEATURES = {}

    @property
    def type(self):
        '''
        The FeatureType of this feature
        '''
        return self._type

    def plot(self, path):
        '''
        Plots the feature into `path`, only creates the file if not implemented
        '''
        #Create empty file if inheriting feature class doesn't define a visualization
        # pylint: disable-next=consider-using-with, disable-next=unspecified-encoding
        open(path, 'a').close()


class Raws(Features):
    '''
    A feature containing the whole raw timeseries as a np.array
    '''
    _type = FeatureType.TIMESERIES

    @staticmethod
    def create(data, max_comps=None, logger=LOGGER):
        '''Create this feature from a DecompData object'''
        if max_comps is not None:
            logger.warn("DEPRECATED: max_comps parameter in features can not guaranty \
sensible choice of components, use n_components parameter for parcellations instead")
        feat = Raws(data.frame, data, data.temporals[:, :, :max_comps])
        return feat

    def flatten(self, feat=None):
        '''
        Flattens the feature into one trial dimension and one dimension for everything else
        '''
        if feat is None:
            feat = self._feature
        return np.reshape(feat, (feat.shape[0], -1))

    @property
    def pixel(self):
        '''
        Creates PixelSlice object, from which slices of recomposed pixel data can be accessed
        the first key is applied to the timepoints, the second and third to the horizontal
        and vertical dimension of the spatials
        '''
        return DecompData.PixelSlice(np.reshape(self._feature, (-1, *self._feature[2:])),
                                     self.data.spatials[:self._feature.shape[2]])

    @property
    def ncomponents(self):
        '''The number of components the data is decomposed into'''
        return self._feature.shape[-1]


#TODO Is this even needed?
class FeatureMean(Features):
    '''
    A derived feature class containing the trial average of another feature
    '''
    def __init__(self, frame, data, feature, base, file=None, full=False):
        super().__init__(frame=frame, data=data, feature=feature, file=file, full=full)
        self._base_feature = base

    @staticmethod
    def create(base):
        '''Create this feature mean from a base feature'''
        feature = np.mean(base.feature, axis=0)[None,:].reshape((1, *base.feature.shape[1:]))
        feat = FeatureMean(base.frame, base.data, feature, base )
        return feat

    def flatten(self, feat=None):
        '''
        Flattens the feature into one trial dimension and one dimension for everything else
        '''
        if feat is None:
            feat = self._feature
        return self._base_feature.flatten(self._feature)

    def save(self, file, data_file=None):
        '''
        not yet implemented
        '''

    @classmethod
    def load(cls, file, data_file=None, feature_hash=None, try_loaded=False):
        '''
        not yet implemented
        '''

    @property
    def ncomponents(self):
        '''The number of components the data is decomposed into'''
        return self._feature.shape[-1]
