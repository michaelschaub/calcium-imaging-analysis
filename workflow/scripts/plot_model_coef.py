import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path
import sys

sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.plotting import draw_neural_activity
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

### Setup
logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    # coefs= np.empty((len(list(snakemake.input))),dtype=object) #perfs of diff. features can have diff. dim
    with open(snakemake.input["model"], "rb") as f:
        pipelines = pickle.load(f) #TODO why is it a list of a list of pipelines and not just a list of pipelines

    classes = np.asarray([pipeline[1].classes_ for pipeline in pipelines[0]]) #n_reps x n_classes 
    print(snakemake.wildcards["feature"])
    
    sort_classes = classes.argsort(axis=1)
    print(classes[0])
    coefs = np.asarray([pipeline[1].coef_ for pipeline in pipelines[0]],dtype=float)# n_reps x n_classes x n_features

    #idx = np.ogrid[tuple(map(slice, classes.shape))]
    #idx[1] = sort_classes
    classes = np.take_along_axis(classes, sort_classes, axis=1)# classes[tuple(idx)]

    #classes = classes[sort_classes]
    #print(classes.shape)
    print(coefs.shape)
    #idx = np.ogrid[tuple(map(slice, coefs.shape))]

    #sort_coefs = np.indices(coefs.shape) # n_reps x n_classes x n_features
    #sort_coefs[]          # n_reps x n_classes
    #np.take_along_axis(B, sort_idxs[:, np.newaxis, :], axis=2)
    #idx[1] = sort_classes

    #sort_idxs = np.argsort(A, axis=1)
    coefs = np.take_along_axis(coefs, sort_classes[:,:,np.newaxis], axis=1)

    #coefs = coefs[tuple(idx)]

    coefs= np.mean(coefs,axis=0) #mean over runs

    spatials = DecompData.load(snakemake.input["parcel"])._spats

    n_comps = spatials.shape[0] 
    n_classes = coefs.shape[0]

    # timepoints x spatials
    # spatials
    # timepoints x spatials x spatials
    # spatials

    try:
        #coefs.reshape((classes, -1, n_comps))  
        logger.info(snakemake.wildcards["feature"])

        coefs.reshape((n_classes, n_comps)) # only works for non_time resolved activity
        #handle feature formatting -> classes x shape here 

        logger.info(coefs.shape)
        logger.info(spatials.shape)

        frames =np.einsum('ik,kjl->ijl',coefs,spatials) #for coef in coefs]
    except Exception as err:
        
        logger.info(err)
        frames=np.mean(spatials,axis=0)
        

    

    draw_neural_activity(frames=frames,path=snakemake.output['coef_plot'],plt_title=f"Coef for {snakemake.wildcards['feature']}",subfig_titles= classes[0],overlay=True,outlined=True, logger=logger)



    snakemake_tools.stop_timer(timer_start, logger=logger)

except Exception:
    logger.exception('')
    sys.exit(1)
