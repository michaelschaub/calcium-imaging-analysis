
import numpy as np
from enum import Enum
from data import DecompData #,Comp Superclass

#functions expert temps in shape of trials x frames x comp with frames being already filtered to same length

#instead of temps, just access temps from comp super class

def mean(temps):
    return temps.mean(1) #average over frames

def covs(temps):
    pass

def corr(temps):
    pass

def autocorrs(temps):
    pass

def autocovs(temps, n_tau=3, T=7):
    trials, frames, comp  = temps.shape
    cov_m = np.zeros([trials, n_tau, comp, comp])

    for trial in range(trials):
        for i_tau in range(n_tau):
            cov_m[trial, i_tau, :, :] = np.tensordot(temps[trial, 0:T - n_tau + 1, :comp],
                                                temps[trial, i_tau:T - n_tau + 1 + i_tau, :comp],
                                                axes=(0, 0)) / float(T - n_tau)

    return cov_m

