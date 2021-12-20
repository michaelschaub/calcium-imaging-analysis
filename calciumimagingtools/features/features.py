import numpy as np
from data import Data, DecompData
from loading import reproducable_hash, load_h5, save_h5

from pymou import MOU
import pathlib
from enum import Enum

#Progress Bar
from tqdm.auto import tqdm

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
        feats = self._feature
        feat = np.empty((data.temporals_flat.shape[0], *feats.shape[1:]), dtype=float)
        starts = list(data._starts)
        starts.append(-1)
        for i in range(len(starts) - 1):
            feat[starts[i]:starts[i + 1]] = feats[i]
        return feat

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
            h5_file, feature = load_h5( file, ["feature"] )
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


class Moup(Features):
    _type = Feature_Type.DIRECTED

    def __init__(self, data, mou_ests, label=None, file=None):
        self.data = data
        self._mou_ests = mou_ests
        self._label = label
        self._savefile = file

    def create(data, max_comps=None, timelag=None, label=None):
        mou_ests = fit_moup(data.temporals[:, :, :max_comps], timelag if timelag>0 else None, label)
        feat = Moup(data, mou_ests, label)
        return feat

    def flatten(self, feat=None):
        n = self._mou_ests[0].get_J().shape[0]
        mask_jac = np.logical_not(np.eye(n, dtype=bool)) # TODO: ADAPT WHEN MASK AVAILABLE
        flat_params = np.empty((len(self._mou_ests), mask_jac.sum()))

        for i,mou_est in enumerate(tqdm(self._mou_ests,desc=self._label,leave=False)):
            flat_params[i,:] = mou_est.get_J()[mask_jac]

        return flat_params

    @property
    def hash(self):
        return reproducable_hash(tuple( getattr(mou,attr)
                                    for attr in Moup.mou_attrs if attr != "d_fit" for mou in self._mou_ests))

    # may need workaround, _feature should be constant and as close to instant access as possible
    # also maybe numpy array
    @property
    def _feature(self):
        return np.asarray([[mou_est.get_J()] for mou_est in self._mou_ests])  # ,other params]

    @property
    def ncomponents(self):
        return self._mou_ests[0].get_J().shape[0]

    mou_attrs = ["n_nodes", "J", "mu", "Sigma", "d_fit"]

    def save(self, file, data_file=None):
        '''
        '''
        attr_arrays = decompose_mou_ests( self._mou_ests )
        attr_arrays["d_fit"] = { key: np.array([ a[key] for a in attr_arrays["d_fit"]]) for key in attr_arrays["d_fit"][0].keys() }

        h5_file = save_h5( self, file, attr_arrays )
        h5_file.attrs["data_hash"] = self.data_hash.hex()
        if self._data.savefile is None:
            if data_file is None:
                path = pathlib.Path(file)
                data_file = path.parent / f"data.{path.stem}{path.suffix}"
            self._data.save(data_file)
        assert (self._data.savefile is not None), "Failure in saving underlaying data object!"
        h5_file.attrs["data_file"] = str(self._data.savefile)
        self._savefile = file

    @classmethod
    def load(Class, file, data_file=None, feature_hash=None, try_loaded=False, label=None):
        if try_loaded and feature_hash is not None and feature_hash in Features.LOADED_FEATURES:
            feat = Features.LOADED_FEATURES[feature_hash]
        else:
            h5_file, *attributes = load_h5( file, labels=Class.mou_attrs )
            if try_loaded and h5_file.attrs["data_hash"] in Data.LOADED_DATA:
                data = Data.LOADED_DATA[h5_file.attrs["data_hash"]]
            elif data_file is None:
                data_file = h5_file.attrs["data_file"]

            attr_arrays = { attr:arr for attr, arr in zip(Class.mou_attrs,attributes) }
            attr_arrays["d_fit"] = [ { k:a for k,a in attr_arrays["d_fit"].items()} for i in range(len(attr_arrays[Moup.mou_attrs[0]])) ]
            mou_ests = recompose_mou_ests(attr_arrays)
            feat = Class(data_file, mou_ests, label)
            feat.data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            Features.LOADED_FEATURES[feat.hash.digest()] = feat
        return feat


def fit_moup(temps, tau, label):
    mou_ests = np.empty((len(temps)),dtype=np.object_)

    for i,trial in enumerate(tqdm(temps,desc=label,leave=False)):
        mou_est = MOU()
        if tau is None:
            raise RuntimeWarning("Moup without lag (integer) given; set i_opt_tau to 1")
            mou_ests[i] = mou_est.fit(trial, i_tau_opt=1, epsilon_C=0.01, epsilon_Sigma=0.01)
        else:
            mou_ests[i] = mou_est.fit(trial, i_tau_opt=tau, epsilon_C=0.01, epsilon_Sigma=0.01) #, regul_C=0.1
            # print number of iterations and model error in log
            print('iter', mou_est.d_fit['iterations'], 'err', mou_est.d_fit['distance'])

        # regularization may be helpful here to "push" small weights to zero here

    return mou_ests

def decompose_mou_ests( mou_ests ):
    attr_arrays = {attr : [] for attr in Moup.mou_attrs}
    for mou in mou_ests:
        for attr in Moup.mou_attrs:
            attr_arrays[attr].append(getattr(mou,attr))
    for attr in Moup.mou_attrs:
        attr_arrays[attr] = np.array(attr_arrays[attr])
    return attr_arrays

def recompose_mou_ests( attr_arrays, mou_ests=None ):
    if mou_ests is None:
        mou_ests = [MOU() for n in attr_arrays[Moup.mou_attrs[0]] ]
    for i, mou in enumerate(mou_ests):
        for attr in Moup.mou_attrs:
            if attr != "d_fit":
                setattr( mou, attr, attr_arrays[attr][i] )
            else:
                mou.d_fit = attr_arrays[attr][i]
    return mou_ests


class Raws(Features):
    _type = Feature_Type.TIMESERIES

    def create(data, max_comps=None):
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


def calc_means(temps):
    return np.mean(temps, axis=1)  # average over frames


class Means(Features):
    _type = Feature_Type.NODE

    def create(data, max_comps=None):
        feat = Means(data, feature=calc_means(data.temporals[:, :, :max_comps]))
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return feat

    @property
    def pixel(self):
        return DecompData.PixelSlice(self._feature, self.data._spats[:self._feature.shape[1]])

    def _op_data(self, a):
        df = self.data._df
        if isinstance(a, Data):
            temps = self.expand(a)
        else:
            temps = self.expand()
        spats = self.data.spatials
        starts = self.data._starts
        return df, temps, spats, starts

    @property
    def ncomponents(self):
        return self._feature.shape[-1]


def calc_covs(temps, means):
    # TODO: Optimize, currently calculates off diagonals double
    temps = temps - means[:, None, :]
    return np.einsum("itn,itm->inm", temps, temps) / temps.shape[1]


def flat_covs(covs, diagonal):
    # true upper triangle matrix (same shape as covariance)
    ind = np.triu(np.ones(covs.shape[1:], dtype=bool),k=int(not diagonal))
    # flattened upper triangle of covariances
    return covs[:, ind]


class Covariances(Features):
    _type = Feature_Type.UNDIRECTED

    def __init__(self, data, feature, means, file=None, include_diagonal=True):
        self.data = data
        self._feature = feature
        self._means = means
        self._savefile = file
        self._include_diagonal = include_diagonal

    def create(data, means=None, max_comps=None, include_diagonal=True):
        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means._feature
        feature = calc_covs(data.temporals[:, :, :max_comps], means)
        feat = Covariances(data, feature, means, include_diagonal=include_diagonal)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return flat_covs(feat, self._include_diagonal)

    def save(self, file, data_file=None):
        '''
        '''
        h5_file = save_h5( self, file, {"feature" : self._feature,
                                        "means" : self._means } )
        h5_file.attrs["data_hash"] = self._data_hash.hex()
        if self._data.savefile is None:
            if data_file is None:
                path = pathlib.Path(file)
                data_file = path.parent / f"data.{path.stem}{path.suffix}"
            self._data.save(data_file)
        assert (self._data.savefile is not None), "Failure in saving underlaying data object!"
        h5_file.attrs["data_file"] = str(self._data.savefile)
        self._savefile = file

    @classmethod
    def load(Class, file, data_file=None, feature_hash=None, try_loaded=False):
        if try_loaded and feature_hash is not None and feature_hash in Features.LOADED_FEATURES:
            feat = Features.LOADED_FEATURES[feature_hash]
        else:
            h5_file, feature, means = load_h5( file, labels=["feature","means"])
            if try_loaded and h5_file.attrs["data_hash"] in Data.LOADED_DATA:
                data = Data.LOADED_DATA[h5_file.attrs["data_hash"]]
            elif data_file is None:
                data_file = h5_file.attrs["data_file"]
            feat = Class(data_file, feature, means, file)
            feat.data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            Features.LOADED_FEATURES[feat.hash.digest()] = feat
        return feat

    @property
    def ncomponents(self):
        return self._feature.shape[-1]


def calc_acovs(temps, means, covs, n_tau_range, label):
    temps = temps - means[:, None, :]
    trials, n_frames, comps = temps.shape
    cov_m = np.zeros([trials, len(n_tau_range)+1, comps, comps])

    cov_m[:, 0] = covs

    for trial in tqdm(range(trials),desc=label,leave=False):

        for i,i_tau in enumerate(n_tau_range,1): #range(1, n_tau + 1):
            cov_m[trial, i, :, :] = np.tensordot(temps[trial, 0:n_frames - i_tau],
                                                     temps[trial, i_tau:n_frames],
                                                     axes=(0, 0)) / float(n_frames - i_tau)
    return cov_m


DEFAULT_TIMELAG = 10


class AutoCovariances(Features):
    _type=Feature_Type.DIRECTED

    def __init__(self, data, feature, means, covs, file=None, include_diagonal=True):
        self.data = data
        self._feature = feature
        self._means = means
        self._covs = covs
        self._savefile = file
        self._include_diagonal = include_diagonal

    def create(data, means=None, covs=None, max_comps=None, max_time_lag=None, time_lag_range=None, label = None, include_diagonal=True):
        if means is None:
            means = calc_means(data.temporals[:, :, :max_comps])
        elif isinstance(means, Means):
            means = means._feature
        if covs is None:
            covs = calc_covs(data.temporals[:, :, :max_comps], means)
        elif isinstance(covs, Covariances):
            covs = np.copy(covs._feature)

        if max_time_lag is None or max_time_lag >= data.temporals.shape[1]:
            max_time_lag = DEFAULT_TIMELAG

        if time_lag_range is None or np.amax(time_lag_range) >= data.temporals.shape[1]:
            time_lag_range = range(1, max_time_lag + 1)

        feature = calc_acovs(data.temporals[:, :, :max_comps], means, covs, time_lag_range, label)
        feat = AutoCovariances(data, feature, means, covs, include_diagonal= include_diagonal)
        return feat

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate((flat_covs(feat[:, 0],diagonal=self._include_diagonal), feat[:, 1:].reshape((feat.shape[0], -1))), axis=1)

    def save(self, file, data_file=None):
        '''
        '''
        h5_file = save_h5( self, file, {"feature" : self._feature,
                                        "means" : self._means,
                                        "covs" : self._covs } )
        h5_file.attrs["data_hash"] = self._data_hash.hex()
        if self._data.savefile is None:
            if data_file is None:
                path = pathlib.Path(file)
                data_file = path.parent / f"data.{path.stem}{path.suffix}"
            self._data.save(data_file)
        assert (self._data.savefile is not None), "Failure in saving underlaying data object!"
        h5_file.attrs["data_file"] = str(self._data.savefile)
        self._savefile = file

    @classmethod
    def load(Class, file, data_file=None, feature_hash=None, try_loaded=False):
        if try_loaded and feature_hash is not None and feature_hash in Features.LOADED_FEATURES:
            feat = Features.LOADED_FEATURES[feature_hash]
        else:
            h5_file, feature, means, covs = load_h5( file, labels=["feature","means","covs"])
            if try_loaded and h5_file.attrs["data_hash"] in Data.LOADED_DATA:
                data = Data.LOADED_DATA[h5_file.attrs["data_hash"]]
            elif data_file is None:
                data_file = h5_file.attrs["data_file"]
            feat = Class(data_file, feature, means, covs, file)
            feat.data_hash = bytes.fromhex(h5_file.attrs["data_hash"])
            Features.LOADED_FEATURES[feat.hash.digest()] = feat
        return feat

    @property
    def ncomponents(self):
        return self._feature.shape[-1]


class FeatureMean(Features):
    def create(base):
        feature = np.mean(base._feature, axis=0).reshape((1, *base._feature.shape[1:]))
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
