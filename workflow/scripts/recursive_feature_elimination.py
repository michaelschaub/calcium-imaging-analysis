import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.feature_selection as skfs


from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.features import Features, Means, Raws, Covariances, AutoCovariances, Moup, Feature_Type
from ci_lib.plotting import graph_circle_plot, construct_rfe_graph, plot_glassbrain_bokeh
from ci_lib import DecompData
from ci_lib.rfe import RFE_pipeline

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","trial_selection","conditions","feature_calculation","decoder"],
                              params=['conds','reps'])
    start = snakemake_tools.start_timer()

    ### Load feature for all conditions
    cond_str = snakemake.params['conds']
    feature = snakemake.wildcards["feature"]
    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "autocovariance" : AutoCovariances, "moup" :Moup }
    feature_class = feature_dict[snakemake.wildcards["feature"].split("_")[0]]

    cond_feats = []
    for path in snakemake.input["feats"]:
        cond_feats.append(feature_class.load(path))

    feat_type = cond_feats[0].type
    rfe_n = snakemake.wildcards["rfe_n"]
    n_rep = 1 #snakemake.params['reps'] #TODO

    ### Scale & Split
    cv = StratifiedShuffleSplit(n_rep, test_size=0.2, random_state=420)

    data = np.concatenate([feat.flatten() for feat in cond_feats])
    labels = np.concatenate([np.full((len(cond_feats[i].flatten())), cond_str[i])
                             for i in range(len(cond_feats))])


    scaler = preprocessing.StandardScaler().fit( data )
    data = scaler.transform(data)
    cv_split = cv.split(data, labels)

    # Build RFE pipeline
    c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=0.00001, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])

    _ , feats = data.shape
    if(rfe_n=="full"):
        rfe_n = feats
    if(int(rfe_n)>int(cond_feats[0].ncomponents) and feat_type == Feature_Type.NODE):
        rfe_n = feats

    RFE = skfs.RFE(c_MLR,n_features_to_select=int(rfe_n))

    ranking = np.zeros([n_rep,feats],dtype=np.int32)
    perf = np.zeros((n_rep))
    decoders = []

    ### Train & Eval
    for i, (train_i, test_i) in enumerate(cv_split):

        RFE.fit(data[train_i, :], labels[train_i])
        ranking[i,:] = RFE.ranking_
        best_feat_iter = np.argsort(ranking[i])[:int(rfe_n)]
        perf[i] = RFE.estimator_.score(data[test_i, :][:,best_feat_iter], labels[test_i])
        decoders.append(RFE.estimator_)

    list_best_feat = np.argsort(ranking.mean(0))[:int(rfe_n)]

    with open(snakemake.output["perf"], 'wb') as f:
        pickle.dump(perf, f)

    with open(snakemake.output["best_feats"], 'wb') as f:
        pickle.dump(list_best_feat, f)

    with open(snakemake.output["model"], 'wb') as f:
        pickle.dump(decoders, f)


    ##Plots

    parcellation = DecompData.load(snakemake.input["parcellation"])
    n_comps = cond_feats[0].ncomponents

    graph_circle_plot(list_best_feat,n_nodes= n_comps, title=feature, feature_type = feat_type, node_labels=parcellation.spatial_labels, save_path=snakemake.output["plot"])

    #Glassbrain Plot
    rfe_graph = construct_rfe_graph(list_best_feat, n_nodes = n_comps, feat_type = feat_type)
    plot_glassbrain_bokeh(graph=rfe_graph,components_spatials=parcellation.spatials,components_labels=parcellation.spatial_labels,save_path=snakemake.output["glassbrain"])
except Exception:
    logger.exception('')
    sys.exit(1)
