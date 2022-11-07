import numpy as np
import pickle

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.features import Features, Means, Raws, Covariances, AutoCovariances, Moup, AutoCorrelations, Cofluctuation
from ci_lib.plotting import graph_circle_plot, plot_glassbrain_bokeh, graph_sping_plot
from ci_lib import DecompData
from ci_lib.networks import construct_network

### Setup
logger = snakemake_tools.start_log(snakemake) # redirect std_out to log file
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    ### Load
    ## TODO: replace with feature class load handler
    feature = snakemake.wildcards["feature"]
    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "autocovariance" : AutoCovariances, "moup" :Moup,"autocorrelation" : AutoCorrelations, "cofluctuation":Cofluctuation }
    feature_class = feature_dict[snakemake.wildcards["feature"].split("_")[0]]
    feat_type = feature_class._type

    print(feature_class)
    print(type(feature_class))
    print(type(feature_class).__name__)
    print(feature)
    print(feat_type)

    cond_feat = feature_class.load(snakemake.input["original_features"][0])
    n_comps = cond_feat.ncomponents

    feats = snakemake_tools.load(snakemake,snakemake.input["features"],dtype='int')


    parcellation = DecompData.load(snakemake.input["parcellation"])
    labels = parcellation.spatial_labels
    cut_labels = labels[:n_comps] if labels is not None else None

    ### Process (Plot)
    ### Save
    print(feats)
    graph_circle_plot(feats,n_nodes= n_comps, title=feature, feature_type = feat_type, node_labels= cut_labels, save_path=snakemake.output["plot"])

    rfe_graph = construct_network(feats, n_nodes = n_comps, feat_type = feat_type)
    plot_glassbrain_bokeh(graph=rfe_graph,components_spatials=parcellation.spatials,components_labels=cut_labels,save_path=snakemake.output["interactive_plot"],small_file=True)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
