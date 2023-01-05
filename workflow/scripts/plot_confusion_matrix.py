import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as cm
import seaborn as sns
import yaml

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()
    with open(snakemake.input["conf_matrix"], "rb") as f:
        conf_matrices = np.asarray(pickle.load(f))

    with open(snakemake.input["norm_conf_matrix"], "rb") as f:
        norm_conf_matrices = np.asarray(pickle.load(f))

    with open(snakemake.input["class_labels"], "rb") as f:
        class_labels = np.asarray(pickle.load(f))

    if len(conf_matrices)>1:
        #Get frame indices for each phase
        phases = snakemake.config["phase_conditions"].copy()
        if "all" in phases:
            phases.pop("all") 
        try:
            phase_ind = [np.arange(int(phase_timings["start"]),int(phase_timings["stop"])-1) for _, phase_timings in phases.items()]

            #Sum absolute number of classifications for each frames resp. mean of relative number over corresponding frames
            conf_matrices = np.asarray([np.mean(conf_matrices[phase_i],axis=0) for phase_i in phase_ind])
            norm_conf_matrices = np.asarray([np.mean(norm_conf_matrices[phase_i],axis=0) for phase_i in phase_ind])
        except Exception as err:
            #Defined phases did not match timepoints of feature, collapse to one timepoint and ignore phases
            conf_matrices = np.asarray([np.mean(conf_matrices,axis=0)])
            norm_conf_matrices = np.asarray([np.mean(norm_conf_matrices,axis=0)])            
  
    parcellation_order = list(snakemake.config["parcellations"].keys()).index(snakemake.wildcards["parcellation"])
    colors = sns.color_palette("dark",n_colors=len(list(snakemake.config["parcellations"].keys())))
    print(colors)
    map_colors = [(1.0,1.0,1.0),colors[parcellation_order]]
    print(map_colors)
    cmap = cm.LinearSegmentedColormap.from_list("my_cmap", map_colors)
    print(cmap(0.5))

    conf_matrices = np.sum(conf_matrices,axis=1).round().astype("int") #sum over runs
    norm_conf_matrices = np.mean(norm_conf_matrices,axis=1) #mean over runs

    
    fig,axs = plt.subplots(1,len(conf_matrices), squeeze=False,sharey=True) #, gridspec_kw={'width_ratios':[1]*len(conf_matrices)})

    c_axs =  fig.add_axes([.91, .3, .015, .4])

    fig.set_figheight(4)
    fig.set_figwidth(3*len(conf_matrices))
    #sns.set(rc={'figure.figsize':(4*len(conf_matrices) , 3.7)})
 
    g = [None]*len(conf_matrices) #np.array((len(conf_matrices)),dtype=object) [None]*

    #if not g.shape:
    #    print(g.shape)
    #    print(len(conf_matrices))
    #    g = [g]

    for i, (conf_matrix, norm_conf_matrix) in enumerate(zip(conf_matrices,norm_conf_matrices)):

        g[i] = sns.heatmap(norm_conf_matrix, cmap=cmap,square=True, vmin=0, vmax=1, annot=conf_matrix, 
                            xticklabels=class_labels, yticklabels=class_labels, fmt='.0f' ,ax=axs[0][i], 
                            cbar=False if i+1<len(conf_matrices) else True, cbar_ax= None if i+1<len(conf_matrices) else c_axs)

             
        g[i].set_xlabel("Predicted Label")
        if conf_matrices.shape[0]>1:
            g[i].set_title(f"{list(phases.keys())[i]}")
        
        if i>0:
        #    g[i].set_yticks([])
        #    g[i].set_ylabel("")
            g[i].tick_params(left=False, bottom=True)
        else:
            #g[i].set_yticks(class_labels)
            g[i].set_ylabel("True Label")
            #g[i].tick_params(left=True, bottom=True)


    #for ax in g:
    #    tl = ax.get_xticklabels()
    #    ax.set_xticklabels(tl, rotation=90)
    #   tly = ax.get_yticklabels()
    #    ax.set_yticklabels(tly, rotation=0)

    #plt.xlabel("Predicted Label")
    #plt.ylabel("True Label")


    feature = snakemake.wildcards["feature"]
    parcellation = snakemake.wildcards["parcellation"]
    title = f"Average confusion matrix of phases for {parcellation}_{feature}"  if len(conf_matrices)>1 else f"Confusion Matrix for {parcellation}_{feature}"
    fig.suptitle(title,wrap=True)

    #fig.tight_layout(rect=[0, 0, .9, 1])

    fig.figure.savefig( snakemake.output[0] )
    fig.figure.savefig( snakemake.output[1] )

    #with open(snakemake.output[0], 'wb') as f:
    #    pickle.dump(fig.figure, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
