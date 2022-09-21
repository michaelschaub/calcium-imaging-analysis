import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.feature_selection as skfs

import networkx as nx


from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.features import Features, Means, Raws, Covariances, AutoCovariances, Moup, AutoCorrelations, Feature_Type
from ci_lib import DecompData
from ci_lib.rfe import RFE_pipeline

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features","decoders"],
                              params=['conds','reps'])
    timer_start = snakemake_tools.start_timer()

    ### Load feature for all conditions
    cond_str = snakemake.params['conds']
    feature = snakemake.wildcards["feature"]
    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "autocovariance" : AutoCovariances, "moup" :Moup,"autocorrelation" : AutoCorrelations }
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
    c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=1.0, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])

    _ , feats = data.shape

    if(float(rfe_n)<=1):
        rfe_n= int(np.round(feats*float(rfe_n)))
    elif(int(rfe_n)>int(cond_feats[0].ncomponents) and feat_type == Feature_Type.NODE):
        rfe_n = feats

    RFE = skfs.RFE(c_MLR,n_features_to_select=int(rfe_n))

    ranking = np.zeros([n_rep,feats],dtype=np.int32)
    perf = np.zeros((n_rep))
    decoders = []

    ### Train & Eval
    for i, (train_i, test_i) in enumerate(cv_split):

        RFE = RFE.fit(data[train_i, :], labels[train_i])
        ranking[i,:] = RFE.ranking_
        best_feat_iter = np.sort(np.argsort(ranking[i])[:int(rfe_n)])
        perf[i] = RFE.estimator_.score(data[test_i, :][:,best_feat_iter], labels[test_i])
        decoders.append(RFE.estimator_)

    list_best_feat = np.argsort(ranking.mean(0))[:int(rfe_n)]

    logger.info("perf")
    logger.info(perf)

    snakemake_tools.save_npy(snakemake,snakemake.output["perf"],perf)


    with open(snakemake.output["best_feats"], 'wb') as f:
        pickle.dump(list_best_feat, f)

    with open(snakemake.output["model"], 'wb') as f:
        pickle.dump(decoders, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
