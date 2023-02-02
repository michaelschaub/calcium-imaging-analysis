import pickle
import numpy as np
import scipy

import os
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as cm
import seaborn as sns
import yaml
import warnings
from sklearn.base import clone

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.plotting import draw_coefs_models
from ci_lib.clustering import plot_DimRed
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

import umap
import hdbscan

logger = start_log(snakemake)

try:
    timer_start = snakemake_tools.start_timer()
    #### Load & Format
    with open(snakemake.input["model"], "rb") as f:
        timepoints_models = pickle.load(f) 
    decomp_object = DecompData.load(snakemake.input["parcel"])

    #only works for models within a sklearn pipeline (including something like scaler and model as second step, hence pipeline[1] = decoder)
    classes = np.asarray([[pipeline[1].classes_ for pipeline in pipelines] for pipelines in timepoints_models]) # timespoints x n_splits x n_classes 
    coefs = np.asarray([[pipeline[1].coef_ for pipeline in pipelines] for pipelines in timepoints_models],dtype=float) # timespoints x n_splits x n_classes x n_features
    intercepts = np.asarray([[pipeline[1].intercept_ for pipeline in pipelines] for pipelines in timepoints_models],dtype=float) 

    #timesteps x runs x class x weight -> (timesteps * runs) x (class * weight) = observations x features
    n_timepoints, n_splits, n_classes, n_weights = coefs.shape
    flat_coefs = np.asarray([ np.squeeze(coef.flatten())
                    for runs in coefs
                    for coef in runs])
    flat_classes = np.asarray([ class_
                    for runs in classes
                    for class_ in runs])
    flat_intercept = np.asarray([ np.squeeze(intercept.flatten())
                    for runs in intercepts
                    for intercept in runs])

    #Order classes in models
    binary = (n_classes==1)
    if not (flat_classes == flat_classes[0,:]).all(-1).any(-1):
        if not binary:
            #Labels are not identically ordered, different order of occurence in reps, sorting needed
            sort_classes = flat_classes.argsort(axis=1)
            flat_classes= np.take_along_axis(flat_classes, sort_classes, axis=1)
            flat_coefs = np.take_along_axis(flat_coefs, sort_classes[:,:,np.newaxis], axis=1)
            flat_intercept = np.take_along_axis(flat_intercept, sort_classes[:,:,np.newaxis], axis=1)
            
        else:
            #For binary case no reordering of coefs possible, invert sign instead
            for i,classes in enumerate(flat_classes):
                #If class order is different from first model (= classes swaped)
                if not (classes == flat_classes[0]):
                    flat_coefs[i] = - flat_coefs[i] #Swap coefs
                    flat_intercept[i] = - flat_intercept[i]

    class_order = flat_classes[0]

    # Compute colors for phases within trial
    phases = snakemake.config["phase_conditions"].copy() 
    if "all" in phases:
        phases.pop("all") 
    n_phases = len(phases)    
    cmap_phases = sns.color_palette("Set2" ,n_colors=n_phases+1,as_cmap=True)
    flat_phase_colors = np.full((n_timepoints*n_splits,4),fill_value=cmap_phases(0))
    flat_coef_phases =  np.full((n_timepoints*n_splits),fill_value=" ",dtype=object)
    for i, (phase_name, phase_timings) in enumerate(phases.items()):
        flat_phase_colors[int(phase_timings["start"])*n_splits:int(phase_timings["stop"])*n_splits,:]=cmap_phases((i+1)/(n_phases+1))
        flat_coef_phases [int(phase_timings["start"])*n_splits:int(phase_timings["stop"])*n_splits] = phase_name


    ###
    #######
    #### Clustering
    #UMAP
    # now we have flat_ coefs,intercept,classes,phase_color
    reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.0,
            n_components=2,
            random_state=42,
    )

    embedding = reducer.fit_transform(flat_coefs)
    fig, ax = plt.subplots(1)
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        alpha=0.05,
        #s=0.5,
        c=flat_phase_colors)
    plt.gca().set_aspect('equal', 'datalim')

    ax.plot(embedding[:, 0], embedding[:, 1],alpha=0.2,color="black") #,linestyle=":")

    #Legend for Phases
    h = [plt.plot([],[], color=cmap_phases((p+1)/(n_phases+1)) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]
    ax.legend(handles=h, labels=list(phases.keys()), title="Trial Phase") #,loc=(-.27,.7),frameon=False)
    #plt.colorbar(fig)
    ax.set_title('UMAP projection of learned models')
    ### Output
    fig.figure.savefig( snakemake.output["no_cluster"] )
    plt.clf()

    ###
    #Cluster
    labels = hdbscan.HDBSCAN(
    min_samples=n_splits,
    min_cluster_size=100,
    ).fit_predict(embedding)

    clustered_points = (labels >= 0)

    clusters = np.unique(labels[clustered_points])

    clusters_inds = [[i for i,label in enumerate(labels) if label == cluster] for cluster in clusters] # results in clusters x ind
    if len(clusters_inds) == 0:
        #No clusters found, combine all models to one single cluster
        clusters_inds = [list(range(0,len(labels)))] 

    #plot clustured
    fig, ax = plt.subplots(1)
    ax.scatter(
        embedding[~clustered_points, 0],
        embedding[~clustered_points, 1],
        color=(0.5, 0.5, 0.5),
        #s=0.1,
        alpha=0.05)
    sc = ax.scatter(
        embedding[clustered_points, 0],
        embedding[clustered_points, 1],
        alpha=0.05,
        #s=0.5,
        c=labels[clustered_points],
        cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')

    ax.plot(embedding[:, 0], embedding[:, 1],alpha=0.2,color="black") #,linestyle=":")

    #Legend for Phases
    #h = [plt.plot([],[], color=cmap_phases((p+1)/(n_phases+1)) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]
    #ax.legend(handles=h, labels=list(phases.keys()), title="Trial Phase") #,loc=(-.27,.7),frameon=False)
    #plt.colorbar(fig)
    ax.set_title('Clustering in UMAP space of learned models')
    ###

    ### Output
    fig.figure.savefig( snakemake.output["cluster"] )
    fig.figure.savefig( snakemake.output["cluster_small"] )


    flat_split_count = np.asarray([np.arange(n_splits)] * n_timepoints).flatten() #0,1,2.... 0,1,2...
    flat_timepoint_count = np.repeat(np.arange(n_timepoints), n_splits) # 0,0,0, ... 1,1,1,
    data_annot= {
        "split":flat_split_count,
        "t":flat_timepoint_count}

    plot_DimRed(flat_coefs,flat_coef_phases, data_annot, path= snakemake.output["cluster_3d"])

    ####
    #### Create model from cluster
    mean_cluster_model = []
    for c,cluster_inds in enumerate(clusters_inds):
        #get arbitrary instance of trained model, copy all static values and overwrite learned coefs and intercept
        new_model = clone(timepoints_models[0][0][1])
        new_model.penalty = timepoints_models[0][0][1].penalty
        new_model.C = timepoints_models[0][0][1].C

        new_model.classes_ =  class_order #timepoints_models[-1][0][1].classes_ #check if classes need to be sorted
        new_model.coef_ = np.mean(flat_coefs[cluster_inds ],axis=(0)).reshape(n_classes, n_weights)
        new_model.intercept_ = np.mean(flat_intercept[cluster_inds ],axis=(0))

        mean_cluster_model.append(new_model)



    with open(snakemake.output['models'], 'wb') as f:
        pickle.dump(mean_cluster_model, f)

    os.mkdir(Path(snakemake.output['coef_plots']))
    for i,model in enumerate(mean_cluster_model):
        path = Path(snakemake.output['coef_plots'])/f"{i}Cluster_coef.pdf"
        path.touch()
        draw_coefs_models([model],decomp_object, snakemake, mean_path=path)
        
    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)