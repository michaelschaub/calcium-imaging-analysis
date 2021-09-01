import numpy as np

class Feature:
    def flatten( self ):
        pass

def calc_means( temps, comps):
    return np.mean(temps,1) #average over frames

class Mean(Feature):
    def __init__( self, data, max_comps=None ):
        self._data = data
        self._feature = calc_means(data.temporals[:,:,:max_comps])
    
    def flatten( self ):
        return self._feature

def calc_covs( temps, means ):
    # TODO: Optimize, currently calculates off diagonals double
    temps = temps - means[:,None,:]
    return np.einsum( "itn,itm, itnm", temps, temps ) / temps.shape[1]

class Covariances(Features):
    def __init__( self, data, means=None, max_comps=None ):
        self._data = data

        if means is None:
            means = calc_means( data[:,:,:max_comps], max_comps )
        elif isinstance( means, Mean ):
            means = means._feature

        self._feature = calc_covs( data.temporals[:,:,:max_comps], means, max_comps )

    def flatten(self):
        pass

def calc_acovs( temps, means, covs, n_tau ):
    temps = temps - means[:,None,:]
    trials, n_frames, comps = temps.shape
    cov_m = np.zeros([trials, n_tau+1, comps, comps])

    cov_m[:,0] = covs
    for trial in range(trials):
        for i_tau in range(1,n_tau+1):
            cov_m[trial, i_tau, :, :] = np.tensordot(temps[trial, 0:n_frames - n_tau + 1],
                                                temps[trial, i_tau:n_frames - n_tau + 1 + i_tau],
                                                axes=(0, 0)) / float(n_frames - n_tau)

    return cov_m

class AutoCovariances(Features):
    def __init__( self, data, means=None, covs=None, max_comps=None, max_time_lag=None ):
        self._data = data

        if covs is None:
            if means is None:
                means = calc_means( data[:,:,:max_comps] )
            elif isinstance( means, Mean ):
                means = means._feature
            covs = calc_covs( data.temporals[:,:,:max_comps], means )
        elif isinstance( acovs, AutoCovariances):
            covs = np.copy(covs.features)

        if max_time_lag is None or max_time_lag >= data.temporals.shape[1]:
            max_time_lag = data.temporals.shape[1]
        self._feature = calc_acovs( data.temporals[:,:,:max_comps], means, covs, max_time_lag )

    def flatten(self):
        pass
