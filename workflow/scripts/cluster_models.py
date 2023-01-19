import pickle
import numpy as np
import scipy

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.colors as cm
import seaborn as sns
import yaml
import warnings

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #### Load & Format
    with open(snakemake.input["model"], "rb") as f:
        timepoints_models = pickle.load(f) #TODO decoding timesteps x n_splits, currently only works for 1 timestep


    logger.info(timepoints_models.shape)
    #only works for models within a sklearn pipeline (including something like scaler and model as second step, hence pipeline[1] = decoder)
    classes = np.asarray([[pipeline[1].classes_ for pipeline in pipelines] for pipelines in timepoints_models]) # timespoints x n_splits x n_classes 
    coefs = np.asarray([[pipeline[1].coef_ for pipeline in pipelines] for pipelines in timepoints_models],dtype=float) # timespoints x n_splits x n_classes x n_features

    n_timepoints, n_splits, n_classes, n_weights = coefs.shape

    logger.info(coefs.shape)
    #timesteps x runs x class x weight -> (timesteps * runs) x (class * weight) = observations x features
    flat_coefs = np.asarray([ np.squeeze(coef.flatten())
                    for runs in coefs
                    for coef in runs])

    flat_classes = np.asarray([ class_
                    for runs in classes
                    for class_ in runs])

    if not (flat_classes == flat_classes[0,:]).all(-1).any(-1):
        #Labels are not identically ordered, different order of occurence in reps, sorting needed
        sort_classes = flat_classes.argsort(axis=1)
        flat_classes= np.take_along_axis(flat_classes, sort_classes, axis=1)
        flat_coefs = np.take_along_axis(flat_coefs, sort_classes[:,:,np.newaxis], axis=1)

    
    logger.info(flat_coefs.shape)
    #### Clustering & plotting
    size = 8

    # Compute colors for phases within trial
    phases = snakemake.config["phase_conditions"].copy() 
    if "all" in phases:
        phases.pop("all") 
    n_phases = len(phases)    
    cmap_phases = sns.color_palette("Set2" ,n_colors=n_phases,as_cmap=True)

    # Map onto timepoints x reps
    #n_timepoints, n_splits = coefs.shape[:2]
    #n_time_reps = len (flat_coefs)
    frame_phase_color = np.full((n_timepoints*n_splits,4),fill_value=cmap_phases(0))
    for i, (phase_name, phase_timings) in enumerate(phases.items()):
        frame_phase_color[int(phase_timings["start"])*n_splits:int(phase_timings["stop"])*n_splits,:]=cmap_phases((i+1)/(n_phases+1))


    vmin, vmax = (np.nanmin(flat_coefs) ,np.nanmax(flat_coefs) )
    norm=mpl.colors.TwoSlopeNorm(vcenter=0,vmin=vmin if vmin<0 else None,vmax=vmax if vmax>0 else None)
    cmap="seismic"
    #Colormap of decoding matches parcellation color
    #parcellation_order = list(snakemake.config["parcellations"].keys()).index(snakemake.wildcards["parcellation"])
    #colors = sns.color_palette("dark",n_colors=len(list(snakemake.config["parcellations"].keys())))
    #map_colors = [(1.0,1.0,1.0),colors[parcellation_order]]
    #cmap_perf = cm.LinearSegmentedColormap.from_list("my_cmap", map_colors)


    sns.set(rc={'figure.figsize':(6 , 6)})

    #No clustering (for comparison)
    fig = sns.clustermap(flat_coefs.transpose(),  figsize=(size , 0.83*size), dendrogram_ratio=0.18, 
                        col_colors=frame_phase_color, norm=norm, cmap=cmap,
                        cbar_pos=(.05, .15, .02, .35),cbar_kws={"label":"Weight"},
                        #xticklabels=framerate, 
                        row_cluster=False,col_cluster=False) 

    #Axis labels
    fig.ax_heatmap.set_yticks(np.arange(0,n_classes)*n_weights+0.5*n_weights) #not supported in clustermap/heatmap, has to be set manually
    fig.ax_heatmap.set_yticklabels(flat_classes[0])
    #fig.ax_heatmap.set_xticks(np.arange(0,n_timepoints)*n_splits+0.5*n_splits) #not supported in clustermap/heatmap, has to be set manually
    #fig.ax_heatmap.set_xticklabels(np.arange(0,n_timepoints))
    fig.ax_heatmap.set(xlabel = f"{n_splits} Splits of {n_timepoints} Frames)",ylabel =f"{n_weights} Weights for {n_classes} Classes)")
    for class_border in np.arange(1,n_classes)*n_weights:
        fig.ax_heatmap.plot([0,  n_timepoints*n_splits], [class_border,class_border], 'w-',alpha=1)
    
    #Legend for Phases
    h = [plt.plot([],[], color=cmap_phases((p+1)/(n_phases+1)) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]
    fig.ax_heatmap.legend(handles=h, labels=list(phases.keys()), title="Trial Phase",loc=(-.27,.7),frameon=False)

    #plot lines for phases
    phases = snakemake.config["phase_conditions"].copy()     # TODO only use phases annotation within chosen phase
    phases.pop("all")
    markers = {}
    for phase_name, phase_timings in phases.items():
        markers[phase_timings["start"]*n_splits] = markers.get(phase_timings["start"]*n_splits, None)
        markers[phase_timings["stop"]*n_splits] = phase_name
    ordered_marker_keys = np.sort(list(markers.keys()))

    for marker in ordered_marker_keys:
        fig.ax_heatmap.plot([marker,marker], [0,  n_weights*n_classes], 'w-',alpha=1)

    fig.figure.savefig( snakemake.output["no_cluster"] )
    plt.clf()


    ############With clustering

    #precompute linkage (to have control over reordering)
    model_linkage = scipy.cluster.hierarchy.linkage(flat_coefs,optimal_ordering=True,method="average",metric='euclidean')

    #cluster = scipy.cluster.hierarchy.fcluster(model_linkage, t=1.15) #.15, criterion='inconsistent', depth=2, R=None, monocrit=None)
    #bkpts = np.where(cluster[:-1] != cluster[1:])[0] +1TODO
    #logger.info(f"Breakpoint: {bkpts}")

    def create_index_tree_recursive(linkage):
        pass 
    #######
    #TODO test recursive perofrmance, space gets massive for many data points
    def create_index_tree(linkage,data):
        #idea from https://stackoverflow.com/questions/9838861/scipy-linkage-format
        clusters = {}
        reorder_data = np.zeros(data.shape[0])
        for i, merge in enumerate(linkage):
            if merge[0] <= len(linkage):
                # if it is an original point read it from the data array
                a = [int(merge[0]) ]
                ha = 0
                # or position in reordered array

                #reorder_data[a]=i
            else:
                # other wise read the cluster that has been created
                a = clusters[int(merge[0])]["children"]
                ha = clusters[int(merge[0])]["height"] + 1

            if merge[1] <= len(linkage):
                b = [int(merge[1]) ]
                hb = 0
                #reorder_data[b]=i
            else:
                b = clusters[int(merge[1])]["children"]
                hb = clusters[int(merge[1])]["height"] + 1

            # the clusters are 1-indexed by scipy
            clusters[1 + i + len(linkage)] = {
                "child_id": (int(merge[0]),int(merge[1])),
                "children": np.concatenate([a, b]),
                "left": a,
                "right": b,
                "height": ha if ha > hb else hb,
            }
        
        #reconstruct ordering, thanks for nothing Sklearn


        logger.info(clusters)
        for i, (id , cluster) in enumerate(clusters.items()):
            a, b = cluster["child_id"]
            if a > len(linkage):
                clusters[a]["parent_id"]= id
            if b > len(linkage):
                clusters[b]["parent_id"]= id


        root=-1
        for i, (id , cluster) in enumerate(clusters.items()):
            if "parent_id" not in cluster:
                root=id
        clusters[root]["parent_id"] = -1

        #todo something like propagate depth
        def propagate(clusters,node,depth,left_start):
            if node <= len(linkage):
                return clusters

            clusters[node]["depth"]=depth
            clusters[node]["left_start"]=left_start

            clusters = propagate(clusters,clusters[node]["child_id"][0],depth+1,left_start)
            offset = len(clusters[clusters[node]["child_id"][0]]["children"]) if clusters[node]["child_id"][0] > len(linkage) else 1
            clusters = propagate(clusters,clusters[node]["child_id"][1],depth+1,left_start+offset)

            return clusters

        clusters = propagate(clusters,root,0,0)


        #uses clusters variable from within this scope
        #but also updates info in clusters so every node is only computed once across all recursions
        '''
        def get_left_bound(cluster,id,rec=0):
            logger.info(f"Recursion start for cluster {id}, recursion {rec}")
            if "left_start" in cluster and "depth" in cluster:
                logger.info(f"depth is already set for {id}, return value")
                return cluster["left_start"],cluster["depth"]
            if cluster["parent_id"] == -1:
                logger.info(f"{id} is root, return number of recursions needed to find root")
                return 0, rec
            parent = clusters[cluster["parent_id"]]
            #logger.info(parent)
            
            if "left_start" in parent:
                logger.info(f"{cluster['parent_id']} of {id} has depth, return value increased by 1")
                #left bound of parent known
                #logger.info(['child_id'])
                if parent['child_id'][0] == id:
                    #Left child of parent
                    logger.info(f"Left child of parent {parent['child_id']}")
                    clusters[id]["left_start"] = parent["left_start"]
                else:
                    #right child of parent
                    logger.info(f"Right child of parent {parent['child_id']}")
                    clusters[id]["left_start"] = parent["left_start"] + len(parent["children"])

                clusters[id]["depth"] = parent["depth"]+1
                return clusters[id]["left_start"],clusters[id]["depth"]
            else: 
                logger.info(f"{cluster['parent_id']} of {id} has no depth, start new recursion with increased recursion count")
                #no left bound for parent known
                #compute left_bound for parent
                clusters[id]["left_start"],clusters[id]["depth"] = get_left_bound(parent,cluster["parent_id"],rec=rec+1)
                if parent['child_id'][1] == id:
                    #if right child, offset by left sibling
                    offset = len(clusters[parent["child_id"][0]]) if parent["child_id"][0] > len(linkage) else 1
                    clusters[id]["left_start"] = clusters[id]["left_start"] + offset

                clusters[id]["depth"] -= 1
                logger.info(clusters[id])
                return clusters[id]["left_start"],clusters[id]["depth"]



            
                cluster[a]["left_start"] = cluster["left_start"]
            
                cluster[a]["left_start"] = cluster["left_start"]
        
        [get_left_bound(cluster,id) for id,cluster in clusters.items()]
        '''

        #ordered_dict = {}
        #for i, _ , cluster in enumerate(clusters.items()):
        #    ordered_dict[np.amin(cluster["children"])]=cluster
            
        ordered_dict = dict(sorted(clusters.items()))
        
        logger.info(ordered_dict)
            # ^ you could optionally store other info here (e.g distances)
        return reorder_data,list(ordered_dict.values())

    reorder, trees=create_index_tree(model_linkage,flat_coefs)
    logger.info(trees)
    #logger.info(trees[-1]["left"])
    logger.info(model_linkage)

    bkpts = []
    for tree in trees:
        logger.info(tree)
        if tree["depth"] <= 3:
            bkpts.append(tree["left_start"])


    #np.cumsum([len(trees[-3]["left"]),len(trees[-3]["right"]),len(trees[-2]["left"])])
    
    #for key,subtree in tree.items():

    
    #clusters = scipy.cluster.hierarchy.fcluster(model_linkage, t=10, criterion='maxclust_monocrit') #, depth=2, R=None, monocrit=None)

    #clusters_ind = np.unique(clusters)
    #clusters_mean_dist = np.zeros((clusters_ind))

    #for cluster in clusters_ind:
    #    cluster_i = np.where(clusters==cluster)

    '''
    max = 20
    min = 4
    n_cluster_range = range(min,max)
    n_clusters = 0

    max_iter = 1000

    dist = scipy.spatial.distance.pdist(flat_coefs,metric='euclidean')
    mean_dist = np.mean(dist)

    t = mean_dist
    t_scale = 0.1
    assumed_magnitude_true_t = -5

    convergence_factor = np.power(np.power(10,assumed_magnitude_true_t),(1/iter))
    bkpts = []
    cluster = []
    converging = False

    while not n_clusters in n_cluster_range:
        cluster = scipy.cluster.hierarchy.fcluster(model_linkage, t=t, criterion='inconsistent', depth=2, R=None, monocrit=None)
   
        n_clusters = len(np.unique(cluster))

        logger.info(f"Threshhold: {t}, N_Clusters: {n_clusters}")
    
        if n_clusters < min:
            t *= (1-t_scale)
            if converging:
                t_scale *= 0.99    
            
        else:
            #Overshot
            converging = True
            t_scale *= 0.99
            t *= (1+t_scale)
            

        max_iter -= 1
        if max_iter<0:
            Warnings.warn("Clustering models did not converge")
            break
    
    bkpts = np.where(cluster[:-1] != cluster[1:])[0] +1
    logger.info(f"{cluster}")
    logger.info(f"Breakpoint: {bkpts}")
    '''


    #logger.info(col_linkage)
    #row_linkage = scipy.cluster.hierarchy.linkage(train_test_performance.transpose(),optimal_ordering=True,method="centroid")

    fig = sns.clustermap(flat_coefs.transpose(), col_colors=frame_phase_color, #cmap =cmap_perf,
                        cbar_pos=(.05, .15, .02, .35),cbar_kws={"label":"Weight"},
                        norm =norm, cmap=cmap, dendrogram_ratio=0.25,
                         
                        col_linkage=model_linkage, row_cluster=False) #row_cluster=False,xticklabels=framerate, yticklabels=framerate,

    #Axis labels
    fig.ax_heatmap.set_yticks(np.arange(0,n_classes)*n_weights+0.5*n_weights) #not supported in clustermap/heatmap, has to be set manually
    fig.ax_heatmap.set_yticklabels(flat_classes[0])
    #fig.ax_heatmap.set_xticks(np.arange(0,n_timepoints)*n_splits+0.5*n_splits) #not supported in clustermap/heatmap, has to be set manually
    #fig.ax_heatmap.set_xticklabels(np.arange(0,n_timepoints))
    fig.ax_heatmap.set(xlabel = f"{n_splits} Splits of {n_timepoints} Frames)",ylabel =f"{n_weights} Weights for {n_classes} Classes)")
    #for class_border in np.arange(1,n_classes)*n_weights:
    #    fig.ax_heatmap.plot([0,  n_timepoints*n_splits], [class_border,class_border], 'w-',alpha=1)
    logger.info(bkpts)
    fig.ax_heatmap.vlines(bkpts, *fig.ax_heatmap.get_ylim(),colors=["black"]*len(bkpts))
    fig.ax_heatmap.hlines(np.arange(1,n_classes)*n_weights, *fig.ax_heatmap.get_xlim(),colors=['black']*(n_classes-1))
    

    #Legend for Phases
    h = [plt.plot([],[], color=cmap_phases((p+1)/(n_phases+1)) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]
    fig.ax_heatmap.legend(handles=h, labels=list(phases.keys()), title="Trial Phase",loc=(-.27,.7),frameon=False)

    #plot lines for clustured breakpoints
    #for bkpt in bkpts:
    #    
    
        #fig.ax_heatmap.plot([bkpt,bkpt], [0,  n_classes* n_weights], 'k-',alpha=1)
        #fig.ax_heatmap.plot([marker,marker], [0,  n_weights], 'w-',alpha=1)

    fig.figure.savefig( snakemake.output["cluster"] )



    



    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
