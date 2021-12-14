import h5py
import numpy
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from matplotlib import pyplot as plt
import itertools

import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.discriminant_analysis as skda
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
import sklearn.ensemble as skens
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit

import scipy
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms

#Progress Bar
#from tqdm.auto import tqdm
#from tqdm.notebook import tqdm
from tqdm import tqdm

from data import DecompData

##better solution?
import sys
sys.path.append(Path(__file__).parent)


from features import Raws,Means, Moup, Covariances, AutoCovariances, Feature_Type
from plotting import graph_circle_plot, plots
from loading import load_task_data_as_pandas_df
from decomposition import anatomical_parcellation


# MLR adapted for recursive feature elimination (RFE)
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self


# add missing h5 files here
missing_task_data = []

### New data extraction
data_path = Path(__file__).parent.parent / Path('resources/experiment')
plot_path = Path(__file__).parent.parent / Path('plots')
if not (data_path/'extracted_data.pkl').exists() :
    # load behavior data
    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(root_paths=[data_path], mouse_ids=["GN06"], reextract=False)
    with open( data_path / 'extracted_data.pkl', 'wb') as handle:
        pkl.dump(sessions, handle)
else:
    # load saved data
    with open( data_path / 'extracted_data.pkl', 'rb') as handle:
        sessions = pkl.load(handle)
    print("Loaded pickled data.")

file_path = data_path / "GN06" / Path('2021-01-20_10-15-16/SVD_data/Vc.mat')
f = h5py.File(file_path, 'r')

frameCnt = np.array(f['frameCnt'])
trial_starts = np.cumsum(frameCnt[:, 1])[:-1]

mask = np.ones( len(trial_starts), dtype=bool )
mask[missing_task_data] = False
trial_starts = trial_starts[mask]


#print(sessions)
#print(frameCnt.shape)
opts_path = data_path/"GN06"/Path('2021-01-20_10-15-16/SVD_data/opts.mat')
trans_params = scipy.io.loadmat(opts_path,simplify_cells=True)['opts']['transParams']

#align_
align_svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts), trans_params=trans_params)
svd = anatomical_parcellation(align_svd)
print(svd.spatial_labels)


#define different conds
modal_keys = ['visual', 'tactile', 'vistact']
modal_range = range(3)

side_keys = ['right', 'left']
side_range = range(2)

#filter for all conds
trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) & (svd.auto_reward == 0) & (svd.both_spouts == 1))

#set condition filter
cond_keys =  itertools.product(side_keys,modal_keys)
cond_keys_str = [f"{s}_{m}" for s, m in list(cond_keys)]

svd = svd[trial_preselection]

print(svd.spatial_labels)
svd.conditions = [ {"modality" : modal, "target_side_left" : side} for side in side_range for modal in modal_range]

print(svd.spatial_labels)


#Hardcoded 'vis_left','tact_right'
#svd.conditions = [(svd.modality == 0) & (svd.target_side_left == 0) & trial_preselection, (svd.modality ==  1) & (svd.target_side_left == 1) & trial_preselection]
#cond_keys_str = ['vis_left','tact_right']

#cond_keys =  list(itertools.product(modal_keys,side_keys))
#cond_keys_str = [f"{s}_{m}" for m, s in cond_keys]

#print(svd.conditions[:,:,:])


#####
save_outputs = True
baseline_mode = None  #### basline mode ('mean' / 'zscore' / None)
comp = 65 ### number componants to use
n_rep = 1  ### number of repetition
n_comp_LDA = None #5  ### number of LDA componants (conds -1)
RFE_edges = 20


#cond_mean = measurements.mean(svd.conditions[0][30:75,:]) #mean of stimulusframes for first cond
#features  = ['mean',"mean(-base)","raw","mou"]
feature_data = {

    "mean": [Means.create(svd.conditions[i,:,30:75],max_comps=comp) for i in range(len(svd.conditions))], #mean of stimulusframes for first cond
    #"mean(-base)": [Means(svd.conditions[i,:,30:75]-Means(svd.conditions[i,:,15:30]),comp) for i in range(len(svd.conditions))],
    #"raw": [Raws(svd.conditions[i,:,30:75],comp) for i in range(len(svd.conditions))], #mean of stimulusframes for first cond,
    "Cov w/o_Diagonal": [Covariances.create(svd.conditions[i,:,30:75],max_comps=comp, include_diagonal= True) for i in tqdm(range(len(svd.conditions)),desc='Conditions')], #mean of stimulusframes for first cond
    #"Cov w/Diagonal": [Covariances(svd.conditions[i,:,30:75],max_comps=comp, include_diagonal= True) for i in tqdm(range(len(svd.conditions)),desc='Conditions')],
    #r"Cov($\tau$=0)": [AutoCovariances(svd.conditions[i,:,30:75],max_comps=comp,time_lag_range=[0]) for i in tqdm(range(len(svd.conditions)),desc='Conditions')],
    #"mou1": [Moup.create(svd.conditions[i,:,30:75],max_comps=comp,time_lag=3) for i in range(len(svd.conditions))],
}


#for j in tqdm(range(1,0,1),desc="Features"):
#    feature_data[r"Cov($\tau$="+str(j)+")"] = [AutoCovariances(svd.conditions[i,:,30:75],max_comps=comp, time_lag_range=[j]) for i in tqdm(range(len(svd.conditions)),desc='Conditions')]
#    feature_data[r'Mou($\tau$='+str(j)+")"] = [Moup(svd.conditions[i,:,30:75],max_comps=comp,time_lag=j) for i in tqdm(range(len(svd.conditions)),desc='Conditions')]


features = list(feature_data.keys())
feature_label = features

#feature_label = ['mean',"mean(stim)-mean(base)","raw","mou"]

cv = StratifiedShuffleSplit(n_rep, test_size=0.2, random_state=420)
perf = np.zeros([n_rep, len(features), 5])
classifiers = {}

##### RFE


######

for i_feat, feat in enumerate(tqdm(features,desc="Training classifiers for each features")):
    print(feature_data[feat][0].type)


    data = np.concatenate([feat.flatten() for feat in feature_data[feat]])
    labels = np.concatenate( [np.full((len(feature_data[feat][i].flatten())),cond_keys_str[i])
                              for i in range(len(feature_data[feat]))] )
    #feature_labels[feat]
    #for i in range(len(feature_data[feat])):
    #= feature_data[feat][i]



    scaler = preprocessing.StandardScaler().fit( data )
    data = scaler.transform(data)

    cv_split = cv.split(data, labels)

    ###### RFE
    _ , feats = data.shape

    rk_inter = np.zeros([n_rep,feats],dtype=np.int_)

    #####



    #c_MLR = skppl.make_pipeline(skppc.StandardScaler(),
    #                            skllm.LogisticRegression(C=1, penalty='l2', multi_class='multinomial',
    #                                                     solver='lbfgs',
    #                                                    max_iter=500))

    c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=0.00001, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])

    RFE_inter = skfs.RFE(c_MLR,n_features_to_select=int(RFE_edges))

    #c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

    #c_LDA = skda.LinearDiscriminantAnalysis(n_components=n_comp_LDA, solver='eigen', shrinkage='auto')

    #c_RF = skens.RandomForestClassifier(n_estimators=100, bootstrap=False)

    #classifiers[feat]={"c_MLR":c_MLR.get_params()['steps'][1][1],"c_1NN":c_1NN,"c_LDA":c_LDA,"c_RF":c_RF}

    i = 0  ## counter
    for train_idx, test_idx in tqdm(cv_split,desc='Fit and Score Classifiers'):
        #print(f'\tRepetition {i:>3}/{n_rep}', end="\r" )
        #c_MLR.fit(data[train_idx, :], labels[train_idx])
        #c_1NN.fit(data[train_idx, :], labels[train_idx])
        #c_LDA.fit(data[train_idx, :], labels[train_idx])
        #c_RF .fit(data[train_idx, :], labels[train_idx])

        RFE_inter.fit(data[train_idx, :], labels[train_idx])
        #RFE_inter.fit(data[train_idx, :][:,list_best_feat], labels[train_idx])


        #perf[i, i_feat, 0] = c_MLR.score(data[test_idx, :], labels[test_idx])
        #perf[i, i_feat, 1] = c_1NN.score(data[test_idx, :], labels[test_idx])
        #perf[i, i_feat, 2] = c_LDA.score(data[test_idx, :], labels[test_idx])
        #perf[i, i_feat, 3] = c_RF.score(data[test_idx, :], labels[test_idx])

        #print("MLR",perf[i, i_feat, 0])


        rk_inter[i,:] = RFE_inter.ranking_
        list_best_feat = np.argsort(rk_inter[i])[:RFE_edges]
        perf[i, i_feat, 4] = RFE_inter.estimator_.score(data[test_idx, :][:,list_best_feat], labels[test_idx])
        #print("RFE MLR",perf[i, i_feat, 4])


        print("ranking",rk_inter)
        print("best_feat",list_best_feat)

        i += 1
    print("best_feat_all",np.argsort(rk_inter.mean(0))[:RFE_edges])
    list_best_feat = np.argsort(rk_inter.mean(0))[:RFE_edges]

    data_path = Path(__file__).parent.parent / Path('resources')
    dict_path = data_path/"meta"/"anatomical.mat"
    dorsal_labels = np.asarray(scipy.io.loadmat(dict_path ,simplify_cells=True) ['areaLabels'], dtype ='str')

    graph_circle_plot(list_best_feat,n_nodes= comp, title=feature_label[i_feat],feature_type = feature_data[feat][0].type,node_labels=svd.spatial_labels)
    #print(f'\tRepetition {n_rep:>3}/{n_rep}' )

'''
if save_outputs:
    np.save('perf_tasks.npy', perf)
plt.figure()
title = ' '.join(["Classifiers Accuracies","for",str(comp),"Components on Condtions:",', '.join(cond_keys_str)]) #str(len(svd.conditions)),"Conditions"])
plt.suptitle(title)


for i, feat in enumerate(tqdm(features,desc="Plotting Features")):
    v1 = plots.colored_violinplot(perf[:, i, 0], positions=np.arange(1) + i - 0.3, widths=[0.1], color="blue")
    v2 = plots.colored_violinplot(perf[:, i, 1], positions=np.arange(1) + i - 0.15, widths=[0.1], color="orange")
    v3 = plots.colored_violinplot(perf[:, i, 2], positions=np.arange(1) + i + 0, widths=[0.1], color="green")
    v4 = plots.colored_violinplot(perf[:, i, 3], positions=np.arange(1) + i + 0.15, widths=[0.1], color="yellow")
    v5 = plots.colored_violinplot(perf[:, i, 4], positions=np.arange(1) + i + 0.3, widths=[0.1], color="grey")
    if i == 0:
        plt.legend( [ v['bodies'][0] for v in [v1,v2,v3,v4,v5]], [ "MLR", "1NN", "LDA","RF","RFE-MLR" ] )



plt.xticks(range(len(features)), [ feat for feat in feature_label ])
plt.plot([-.5, len(features)-.5], [1/len(svd.conditions), 1/len(svd.conditions)], '--k')
plt.yticks(np.arange(0, 1, 0.1))
plt.ylabel('Accuracy', fontsize=14)

plt.savefig( plot_path/(title+".png") )
'''

### Plots LDA
'''
for i, feat in enumerate(classifiers):
    if feat in ["raw","mou"]:
        continue
    for classifier in ["c_LDA","c_MLR"]:
        conditions = classifiers[feat][classifier].classes_
        plots.plot_frame(classifiers[feat][classifier].coef_, svd.spatials[:comp,:,:], conditions, plot_path/("Coef of "+classifier+" for Feat: "+feature_label[i])) ##comp = number of components , weights.shape = _ , comp
        #plots.plot_frame(classifiers[feat][classifier].means_, svd.spatials[:comp,:,:], conditions, "Means of "+classifier+" for Feat: "+feature_label[i])

        plots.plot_frame(classifiers[feat][classifier].coef_[[1,4]]-classifiers[feat][classifier].coef_[[2,5]], svd.spatials[:comp,:,:], ["vistact_left - vis_left","vistact_right - vis_right"], plot_path/("Difference of Coef from "+classifier+" for Feat: "+feature_label[i]))
'''
plt.show()


### Show LDA weights
#trial_preselection = ((svd.n_targets == 6) & (svd.n_distractors == 0) &
#                      (svd.auto_reward == 0) & (svd.both_spouts == 1))
#Vc = .temporals_flat ##can'T do that without filtering
#Vc_mean = svd[30:75,trial_preselection].temporals_flat.mean(axis=0)
#Vc_baseline_mean = svd[15:30,trial_preselection].temporals_flat.mean(axis=0)
#weights = c_LDA.coef_#.means_ #coef_

#weights = svd[:6,0].temporals #c_LDA.coef_ #c_LDA.means_ #
#weights = c_LDA.coef_
#conditions = c_LDA.classes_
#print(conditions)
#plots.plot_frame(c_LDA.coef_, svd.spatials[:comp,:,:], conditions, "Coef of Classifier (LDA) for") ##comp = number of components , weights.shape = _ , comp

#plots.plot_frame(c_LDA.means_, svd.spatials[:comp,:,:], conditions)
