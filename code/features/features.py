import numpy as np
from data import Data, DecompData

class Features:
    # flatten contained feauture to one trial and one feature dimension
    def flatten( self ):
        pass

    # expand feature into same shape as temporals in Data (for computation)
    def expand( self, data=None ):
        if data is None:
            data = self._data
        feats = self._feature
        feat = np.empty( (data.temporals_flat.shape[0], *feats.shape[1:] ), dtype=float )
        starts = list(data._starts)
        starts.append(-1)
        for i in range(len(starts)-1):
            feat[starts[i]:starts[i+1]] = feats[i]
        return feat

    @property
    def mean( self ):
        return FeatureMean( self )

class Raws(Features):
    def __init__( self, data, max_comps=None ):
        self._data = data
        self._feature = data.temporals[:,:,:max_comps]

    def flatten( self, feat=None ):
        if feat is None:
            feat = self._feature
        return np.reshape(feat, (feat.shape[0], -1))

    @property
    def pixel( self ):
        return DecompData.PixelSlice( self.reshape(self._feature, (-1, *self._feature[2:])), self._data._spats[:self._feature.shape[2]] )

def calc_means( temps ):
    return np.mean(temps, axis=1) #average over frames

class Means(Features):
    def __init__( self, data, max_comps=None ):
        self._data = data
        self._feature = calc_means(data.temporals[:,:,:max_comps])
    
    def flatten( self, feat=None):
        if feat is None:
            feat = self._feature
        return feat

    @property
    def pixel( self ):
        return DecompData.PixelSlice( self._feature, self._data._spats[:self._feature.shape[1]] )

    def _op_data(self, a):
        df = self._data._df
        if isinstance(a, Data):
            temps = self.expand(a)
        else:
            temps = self.expand()
        spats = self._data.spatials
        starts = self._data._starts
        return df, temps, spats, starts

def calc_covs( temps, means ):
    # TODO: Optimize, currently calculates off diagonals double
    temps = temps - means[:,None,:]
    return np.einsum( "itn,itm->inm", temps, temps ) / temps.shape[1]

def flat_covs( covs ):
    # true upper triangle matrix (same shape as covariance)
    ind = np.triu( np.ones( covs.shape[1:], dtype=bool ) )
    # flattened upper triangle of covariances
    return covs[:, ind]

class Covariances(Features):
    def __init__( self, data, means=None, max_comps=None ):
        self._data = data

        if means is None:
            self._means = calc_means( data.temporals[:,:,:max_comps] )
        elif isinstance( means, Means ):
            self._means = means._feature
        else:
            self._means = mean

        self._feature = calc_covs( data.temporals[:,:,:max_comps], self._means )

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return flat_covs( feat )

def calc_acovs( temps, means, covs, n_tau ):
    temps = temps - means[:,None,:]
    trials, n_frames, comps = temps.shape
    cov_m = np.zeros([trials, n_tau+1, comps, comps])

    cov_m[:,0] = covs
    for trial in range(trials):
        for i_tau in range(1,n_tau+1):
            cov_m[trial, i_tau, :, :] = np.tensordot(temps[trial, 0:n_frames - i_tau],
                                                temps[trial, i_tau:n_frames],
                                                axes=(0, 0)) / float(n_frames - i_tau)
    return cov_m

DEFAULT_TIMELAG = 10

class AutoCovariances(Features):
    def __init__( self, data, means=None, covs=None, max_comps=None, max_time_lag=None ):
        self._data = data

        if means is None:
            self._means = calc_means( data.temporals[:,:,:max_comps] )
        elif isinstance( means, Means ):
            self._means = means._feature
        else:
            self._means = means
        if covs is None:
            self._covs = calc_covs( data.temporals[:,:,:max_comps], self._means )
        elif isinstance( covs, Covariances):
            self._covs = np.copy(covs._feature)
        else:
            self._covs = covs

        if max_time_lag is None or max_time_lag >= data.temporals.shape[1]:
            max_time_lag = DEFAULT_TIMELAG
        self._feature = calc_acovs( data.temporals[:,:,:max_comps], self._means, self._covs, max_time_lag )

    def flatten(self, feat=None):
        if feat is None:
            feat = self._feature
        return np.concatenate( (flat_covs(feat[:,0]), feat[:,1:].reshape((feat.shape[0], -1)) ), axis=1 )

class FeatureMean(Features):
    def __init__(self, base ):
        self._base_feature = base
        self._data = base._data
        self._feature = np.mean( base._feature, axis=0 ).reshape((1, *base._feature.shape[1:]) )

    def flatten( self, feat=None):
        if feat is None:
            feat = self._feature
        return self._base_feature.flatten( self._feature )

    @property
    def pixel( self ):
        if isinstance( self._base_feature, Raws ):
            return DecompData.PixelSlice( np.reshape( self._feature, (self._feature.shape[1:])),
                            self._data._spats[:self._feature.shape[2]] )
        elif isinstance( self._base_feature, Means ):
            return DecompData.PixelSlice( self._feature,
                            self._data._spats[:self._feature.shape[1]] )
        else:
            raise AttributeError
