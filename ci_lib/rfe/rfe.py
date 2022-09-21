import numpy as np



from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.feature_selection as skfs


import networkx as nx

from ci_lib.features import Features, Means, Raws, Covariances, AutoCovariances, Moup, AutoCorrelations, Feature_Type


def rfe(feat_type, cond_feats, cond_str, n_rep, rfe_n):

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

    print("rfe_n",rfe_n)
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


    return