import pickle
import numpy as np
import scipy

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
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
    with open(snakemake.input["perf"], "rb") as f:
        perf_matrix = np.asarray(pickle.load(f))

    framerate=15 #TODO replace by config entry

    sns.set(rc={'figure.figsize':(9 , 7.5)})

    train_test_performance = np.mean(perf_matrix,axis=2) #.transpose()

    #Create color map for the phase of each frame
    phases = snakemake.config["phase_conditions"].copy() 

    if "all" in phases:
        phases.pop("all") 
    n_phases = len(phases)    
    cmap = sns.color_palette("twilight",n_colors=n_phases,as_cmap=True)
    

    n_frames = len(train_test_performance)   
    frame_phase_color = np.full((n_frames,4),fill_value=cmap(1))
    for i, (phase_name, phase_timings) in enumerate(phases.items()):
        frame_phase_color[int(phase_timings["start"]):int(phase_timings["stop"]),:]=cmap(i/n_phases)

    if len(train_test_performance)>1:
        
        #precompute linkage (to avoid reordering)
        col_linkage = scipy.cluster.hierarchy.linkage(train_test_performance,optimal_ordering=True,method="centroid")
        row_linkage = scipy.cluster.hierarchy.linkage(train_test_performance.transpose(),optimal_ordering=True,method="centroid")


        fig = sns.clustermap(train_test_performance,  row_colors=frame_phase_color, col_colors=frame_phase_color, cmap = plt.cm.Blues,square=True, vmin = 0, vmax = 1,col_linkage=col_linkage,row_linkage=row_linkage) #row_cluster=False,xticklabels=framerate, yticklabels=framerate,
        fig.ax_heatmap.set(xlabel = "t (test)",ylabel ="t (train)")

        #second legend
        #leg = plt.legend(loc=(1.03,0), title="Year")
        #fig.ax_heatmap.add_artist(leg)
        h = [plt.plot([],[], color=cmap(p/n_phases) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]

        fig.ax_heatmap.legend(handles=h, labels=list(phases.keys()), title="True Phase")

        fig.figure.savefig( snakemake.output["cluster"] )
    

    else:
        fig = sns.heatmap(train_test_performance ,  cmap = plt.cm.Blues,square=True, vmin =0, vmax = 1,xticklabels=framerate, yticklabels=framerate)
        fig.figure.savefig( snakemake.output["cluster"] )

    plt.clf()
    #Only one timepoint, no clustering possible
    #fig = sns.heatmap(train_test_performance ,  cmap = plt.cm.Blues,square=True, vmin =0, vmax = 1,xticklabels=framerate, yticklabels=framerate)

    size = 8
    #plt.subplots_adjust(right=0.75)
    fig = sns.clustermap(train_test_performance,  figsize=(size , 0.83*size), dendrogram_ratio=(0.17,0.001),
                        row_colors=frame_phase_color, col_colors=frame_phase_color, cmap = plt.cm.Blues,square=True, cbar_pos=(.05, .15, .02, .4),cbar_kws={"label":"Decoding Performance"},
                        vmin = 0, vmax = 1,xticklabels=framerate, yticklabels=framerate,
                        row_cluster=False,col_cluster=False) 
    
    
    h = [plt.plot([],[], color=cmap(p/n_phases) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]

    fig.ax_heatmap.legend(handles=h, labels=list(phases.keys()), title="Trial Phase",loc=(-.27,.7),frameon=False)
    #norm = mpl.colors.Normalize(vmin=0, vmax=1)
    #fig.ax_heatmap.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues),cax=fig.ax_heatmap, orientation='vertical', label='Decoding Perfromance')
    fig.ax_heatmap.set(xlabel = "t (test)",ylabel ="t (train)")

    fig.figure.savefig( snakemake.output[0] )
    fig.figure.savefig( snakemake.output[1] )

    with open(snakemake.output["model"], 'wb') as f:
        pickle.dump(fig.figure, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
