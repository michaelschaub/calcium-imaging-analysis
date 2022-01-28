import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

import scipy
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms
import warnings

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"calciumimagingtools").absolute()))

from utils import snakemake_tools
from features import Features, Means, Raws, Covariances, AutoCovariances, Moup, Feature_Type
from plotting import graph_circle_plot
from data import DecompData



# MLR adapted for recursive feature elimination (RFE)
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self


# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","prefilters","conditions","feature_calculation","decoder"],
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
n_rep = 1 #snakemake.params['reps']

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

data = DecompData.load(snakemake.input["labels"])
graph_circle_plot(list_best_feat,n_nodes= cond_feats[0].ncomponents, title=feature, feature_type = feat_type, node_labels=data.spatial_labels, save_path=snakemake.output["plot"])
