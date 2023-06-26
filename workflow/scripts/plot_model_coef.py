import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path
import sys

sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.plotting import draw_neural_activity, cmap_blueblackred
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

### Setup
logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    with open(snakemake.input["model"], "rb") as f:
        pipelines = pickle.load(f) #TODO decoding timesteps x n_reps, currently only works for 1 timestep

    classes = np.asarray([pipeline[1].classes_ for pipeline in pipelines[0]]) #n_reps x n_classes 
    coefs = np.asarray([pipeline[1].coef_ for pipeline in pipelines[0]],dtype=float) # n_reps x n_classes x n_features

    if not (classes == classes[0,:]).all(-1).any(-1):
        #Labels are not identically ordered, different order of occurence in reps, sorting needed
        sort_classes = classes.argsort(axis=1)
        classes = np.take_along_axis(classes, sort_classes, axis=1)
        coefs = np.take_along_axis(coefs, sort_classes[:,:,np.newaxis], axis=1)
    
    dispersion = np.std(coefs,axis=0)
    logger.info(np.amax(dispersion))
    coefs= np.mean(coefs,axis=0) #mean over runs
    
    spatials = DecompData.load(snakemake.input["parcel"])._spats

    n_comps = spatials.shape[0] 
    n_classes = coefs.shape[0]

    #Handle different types of features
    # timepoints x spatials
    # spatials
    # timepoints x spatials x spatials
    # spatials
    
    try:
        #coefs.reshape((classes, -1, n_comps))  
        logger.info(snakemake.wildcards["feature"])

        coefs.reshape((n_classes, n_comps)) # only works for non_time resolved activity
        dispersion.reshape((n_classes, n_comps))

        #handle feature formatting -> classes x shape here 

        means =np.einsum('ik,kjl->ijl',coefs,spatials) #for coef in coefs]
        dispersion = np.einsum('ik,kjl->ijl',dispersion,spatials)
    except Exception as err:
        print(f"Plotting feature type {snakemake.wildcards['feature']} currently not supported")
        logger.info(err)
        means=np.mean(spatials,axis=0)
        dispersion =np.std(spatials,axis=0)
        
    labels=classes[0] if n_classes>1 else classes[0][1] #n-classes or binary case, where last class corresponds to 1
    
    draw_neural_activity(frames=means,path=snakemake.output['coef_plot'],plt_title=f"Mean Coef for {snakemake.wildcards['feature']} across Splits",subfig_titles= labels,overlay=True,outlined=True, logger=logger,font_scale=snakemake.config["font_scale"])
    draw_neural_activity(frames=dispersion,path=snakemake.output['var_plot'],plt_title=f"Std Coef for {snakemake.wildcards['feature']} across Splits",subfig_titles= labels,overlay=True,outlined=True, logger=logger,font_scale=snakemake.config["font_scale"])

    snakemake_tools.stop_timer(timer_start, logger=logger)

except Exception:
    logger.exception('')
    sys.exit(1)
