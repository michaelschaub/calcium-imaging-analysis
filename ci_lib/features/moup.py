import numpy as np

from ci_lib import Data, DecompData
from ci_lib.loading import reproducable_hash, load_h5, save_h5
from ci_lib.networks import MOU #from pymou import MOU

import pathlib
from enum import Enum

import logging
LOGGER = logging.getLogger(__name__)

from .features import Features, Feature_Type


def fit_moup(temps, tau, label, logger=LOGGER):
    mou_ests = np.empty((len(temps)),dtype=np.object_)

    for i,trial in enumerate(temps):
        mou_est = MOU()
        if tau is None:
            raise RuntimeWarning("Moup without lag (integer) given; set i_opt_tau to 1")
            mou_ests[i] = mou_est.fit(trial, i_tau_opt=1, epsilon_C=0.01, epsilon_Sigma=0.01)
        else:
            mou_ests[i] = mou_est.fit(trial, i_tau_opt=tau, epsilon_C=0.01, epsilon_Sigma=0.01) #, regul_C=0.1
            # print number of iterations and model error in log
            logger.info('iter {}, err {}'.format( mou_est.d_fit['iterations'], mou_est.d_fit['distance']))

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


class Moup(Features):
    _type = Feature_Type.DIRECTED

    def __init__(self, data, mou_ests, label=None, file=None):
        self.data = data
        self._mou_ests = mou_ests
        self._label = label
        self._savefile = file

    def create(data, max_comps=None, timelag=None, label=None, logger=LOGGER):
        mou_ests = fit_moup(data.temporals[:, :, :max_comps], timelag if timelag>0 else None, label, logger=logger)
        feat = Moup(data, mou_ests, label)
        return feat

    def flatten(self, feat=None):
        n = self._mou_ests[0].get_J().shape[0]
        mask_jac = np.logical_not(np.eye(n, dtype=bool)) # TODO: ADAPT WHEN MASK AVAILABLE
        flat_params = np.empty((len(self._mou_ests), mask_jac.sum()))

        for i,mou_est in enumerate(self._mou_ests):
            flat_params[i,:] = mou_est.get_J()[mask_jac]

        return flat_params

    @property
    def hash(self):
        return reproducable_hash(tuple( getattr(mou,attr)
                                    for attr in Moup.mou_attrs if attr != "d_fit" for mou in self._mou_ests))

    @property
    def feature(self):
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
