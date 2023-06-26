import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as cm
import seaborn as sns
import yaml
import os

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent.parent).absolute()))

from ci_lib.plotting import draw_neural_activity
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #Loading
    decomps = []
    temps = []
    

    for decomp_data_path in snakemake.input:
        decomps.append(DecompData.load(decomp_data_path))
        temps.append(decomps[-1].temporals)

    n_trials, n_frames, n_comps = temps[0].shape
    n_conds = len(decomps)

    os.mkdir(Path(snakemake.output[0]))
    
    #More than 1 timepoint
    
    if (n_frames > 1):
        phases = snakemake.config["phase_conditions"].copy()
        if "all" in phases:
            phases.pop("all") 

        #Get frame indices for each phase
        phase_names = list(phases.keys())

        phase_ind = [np.arange(int(phase_timings["start"]),int(phase_timings["stop"])-1) for _, phase_timings in phases.items()]

        phase_temps = [[ np.asarray(temps[cond])[:,phase_i,:]  for cond in range(n_conds)] for phase_i in phase_ind]

        #Sum absolute number of classifications for each frame resp. mean of relative number over corresponding frames


         
    for phase_temp, phase_name in zip(phase_temps,phase_names):


        cond_mean_temp = np.zeros((n_conds,n_comps))
        cond_labels = list(snakemake.params["conds"].keys())
        logger.info(cond_labels)

        for i,temp in enumerate(phase_temp):
            logger.info(temp.shape)
            cond_mean_temp[i]= np.mean(temp,axis=(0,1)) #mean over trials and frames

        cond_diff_matrix = np.zeros((n_conds,n_conds,n_comps))
        
        spatials = decomps[0].spatials
        _ , width, height = spatials.shape
        cond_diff_frame = np.zeros((n_conds,n_conds,width,height))
        label_matrix = np.zeros((n_conds,n_conds),dtype=object)

        #Compute differences and plot results
        #fig, axes = plt.subplots(rows=n_conds, cols=n_conds, shared_xaxes=True, shared_yaxes=True, column_titles = cond_labels, row_titles = cond_labels , column_widths = [500]*(n_conds) , row_heights = [500]*(n_conds))


        for x,t1 in enumerate(cond_mean_temp):
            for y,t2 in enumerate(cond_mean_temp):
                if x==y:
                    cond_diff_matrix[x,y,:] = cond_mean_temp[x,:] 
                    label_matrix [x,y] = f"Mean {cond_labels[x]}"
                    cond_diff_frame[x,y,:,:] = np.einsum('k,kjl->jl',cond_diff_matrix[x,y,:],spatials) 
                else:
                    if x<y:
                        cond_diff_matrix[x,y,:] = np.subtract(cond_mean_temp[x,:], cond_mean_temp[y,:])
                        label_matrix [x,y] = f"AbsDiff: {cond_labels[x]} − {cond_labels[y]}"
                        cond_diff_frame[x,y,:,:] = np.absolute(np.einsum('k,kjl->jl',cond_diff_matrix[x,y,:],spatials))
                    else:
                        cond_diff_matrix[x,y,:] = np.subtract(cond_mean_temp[x,:], cond_mean_temp[y,:])
                        label_matrix [x,y] = f"Diff: {cond_labels[x]} − {cond_labels[y]}"
                        cond_diff_frame[x,y,:,:] = np.einsum('k,kjl->jl',cond_diff_matrix[x,y,:],spatials) 

                

        #Flatten matrix
        #TODO draw_neural_activity should support 2D matrix of frames (currently only 0D and 1D)
        flat_cond_frames = np.reshape(cond_diff_frame,(-1,width,height))
        flat_labels = label_matrix.flatten()

        path = f"{snakemake.output[0]}/{snakemake.wildcards['parcellation']}_{phase_name}_cond_difference_b-maps.pdf"
        Path(path).touch()
        draw_neural_activity(frames=flat_cond_frames,path=path,plt_title=phase_name,subfig_titles=flat_labels,overlay=True,outlined=True,masked=True,logger=logger,share_v=True, vmin=None,vmax=None,font_scale=1)


    Path(snakemake.output["static"]).touch()
    draw_neural_activity(frames=flat_cond_frames,path=snakemake.output["static"],plt_title="",subfig_titles=flat_labels,overlay=True,outlined=True,masked=True,logger=logger,share_v=True, vmin=None,vmax=None,font_scale=1)
    with open(snakemake.output["static"], 'w') as fp:
        pass
    
    
    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
