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


from data import DecompData

##better solution?
import sys
sys.path.append(Path(__file__).parent)


from features import measurements
from plotting import plots
from loading import load_task_data_as_pandas_df






### New data extraction
data_path = Path(__file__).parent.parent / Path('data')
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
trial_starts = np.cumsum(frameCnt[:, 1])
svd = DecompData( sessions, np.array(f["Vc"]), np.array(f["U"]), np.array(trial_starts) )


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
svd.conditions = ([(svd.modality == modal) & (svd.target_side_left == side) & trial_preselection for side,modal in itertools.product(side_range,modal_range)])
print(cond_keys_str)

#Hardcoded 'vis_left','tact_right'
#svd.conditions = [(svd.modality == 0) & (svd.target_side_left == 0) & trial_preselection, (svd.modality ==  1) & (svd.target_side_left == 1) & trial_preselection]
#cond_keys_str = ['vis_left','tact_right']

#cond_keys =  list(itertools.product(modal_keys,side_keys))
#cond_keys_str = [f"{s}_{m}" for m, s in cond_keys]

#print(svd.conditions[:,:,:])


#####
save_outputs = True
frames = 30  #### trial duration
baseline_mode = None  #### basline mode ('mean' / 'zscore' / None)
runs = 200  ### number of runs
comp = 1 ### number componants to use
n_rep = 10  ### number of repetition
n_comp_LDA = None #5  ### number of LDA componants (conds -1)


#cond_mean = measurements.mean(svd.conditions[0][30:75,:]) #mean of stimulusframes for first cond
features  = ['mean',"mean(-base)"]
feature_data = {
    "mean": [measurements.mean(svd.conditions[i][30:75,:,:],comp) for i in range(len(svd.conditions))], #mean of stimulusframes for first cond
    "mean(-base)": [measurements.mean(svd.conditions[i][30:75,:,:],comp)-measurements.mean(svd.conditions[i][15:30,:,:],comp) for i in range(len(svd.conditions))]
}
feature_label = ['mean',"mean(stim)-mean(base)"]

cv = StratifiedShuffleSplit(n_rep, test_size=0.2, random_state=420)
perf = np.zeros([n_rep, len(features), 4])
classifiers = {}

for i_feat, feat in enumerate(features):


    data = np.concatenate(feature_data[feat])
    labels = np.concatenate([np.full((len(feature_data[feat][i])),cond_keys_str[i]) for i in range(len(feature_data[feat]))])
    #feature_labels[feat]
    #for i in range(len(feature_data[feat])):
         #= feature_data[feat][i]



    scaler = preprocessing.StandardScaler().fit( data )
    data = scaler.transform(data)

    cv_split = cv.split(data, labels)

    c_MLR = skppl.make_pipeline(skppc.StandardScaler(),
                                skllm.LogisticRegression(C=1, penalty='l2', multi_class='multinomial',
                                                         solver='lbfgs',
                                                         max_iter=500))
    c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

    c_LDA = skda.LinearDiscriminantAnalysis(n_components=n_comp_LDA, solver='eigen', shrinkage='auto')

    c_RF = skens.RandomForestClassifier(n_estimators=100, bootstrap=False)

    classifiers[feat]={"c_MLR":c_MLR.get_params()['steps'][1][1],"c_1NN":c_1NN,"c_LDA":c_LDA,"c_RF":c_RF}

    i = 0  ## counter
    for train_idx, test_idx in cv_split:
        print(f'\tRepetition {i:>3}/{n_rep}', end="\r" )
        c_MLR.fit(data[train_idx, :], labels[train_idx])
        c_1NN.fit(data[train_idx, :], labels[train_idx])
        c_LDA.fit(data[train_idx, :], labels[train_idx])
        c_RF .fit(data[train_idx, :], labels[train_idx])
        perf[i, i_feat, 0] = c_MLR.score(data[test_idx, :], labels[test_idx])
        perf[i, i_feat, 1] = c_1NN.score(data[test_idx, :], labels[test_idx])
        perf[i, i_feat, 2] = c_LDA.score(data[test_idx, :], labels[test_idx])
        perf[i, i_feat, 3] = c_RF.score(data[test_idx, :], labels[test_idx])
        i += 1
    print(f'\tRepetition {n_rep}/{n_rep}' )

if save_outputs:
    np.save('perf_tasks.npy', perf)
plt.figure()
title = ' '.join(["Classifiers Accuracies","for",str(comp),"Components on Condtions: ",', '.join(cond_keys_str)]) #str(len(svd.conditions)),"Conditions"])
plt.suptitle(title)
for i, feat in enumerate(features):
    v1 = plots.colored_violinplot(perf[:, i, 0], positions=np.arange(1) + i - 0.3, widths=[0.15], color="blue")
    v2 = plots.colored_violinplot(perf[:, i, 1], positions=np.arange(1) + i - 0.1, widths=[0.15], color="orange")
    v3 = plots.colored_violinplot(perf[:, i, 2], positions=np.arange(1) + i + 0.1, widths=[0.15], color="green")
    v4 = plots.colored_violinplot(perf[:, i, 3], positions=np.arange(1) + i + 0.3, widths=[0.15], color="yellow")
    if i == 0:
        plt.legend( [ v['bodies'][0] for v in [v1,v2,v3,v4]], [ "MLR", "1NN", "LDA","RF" ] )



plt.xticks(range(len(features)), [ feat for feat in feature_label ])
plt.plot([-.5, len(features)-.5], [1/len(svd.conditions), 1/len(svd.conditions)], '--k')
plt.yticks(np.arange(0, 1, 0.1))
plt.ylabel('Accuracy', fontsize=14)

plt.savefig(title+".png")


### Plots LDA

for i, feat in enumerate(classifiers):
    for classifier in ["c_LDA","c_MLR"]:
        conditions = classifiers[feat][classifier].classes_
        plots.plot_frame(classifiers[feat][classifier].coef_, svd.spatials[:comp,:,:], conditions, "Coef of "+classifier+" for Feat: "+feature_label[i]) ##comp = number of components , weights.shape = _ , comp
        #plots.plot_frame(classifiers[feat][classifier].means_, svd.spatials[:comp,:,:], conditions, "Means of "+classifier+" for Feat: "+feature_label[i])

        plots.plot_frame(classifiers[feat][classifier].coef_[[1,4]]-classifiers[feat][classifier].coef_[[2,5]], svd.spatials[:comp,:,:], ["vistact_left - vis_left","vistact_right - vis_right"], "Difference of Coef from "+classifier+" for Feat: "+feature_label[i])

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