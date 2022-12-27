import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.discriminant_analysis as skda
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
import sklearn.ensemble as skens
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle

#Hide convergene warning for shuffled data
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup, Cofluctuation


### Select decoder
# TODO as classes
def MLR(cores=1):
    return skppl.make_pipeline(skppc.StandardScaler(),
                                skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial',
                                                            solver='lbfgs', max_iter=2000, n_jobs=cores, warm_start=True))

def NN(cores):
    return sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

def LDA(cores):
    return skda.LinearDiscriminantAnalysis(n_components=None, solver='eigen', shrinkage='auto')

def RF(cores):
    return skens.RandomForestClassifier(n_estimators=10, bootstrap=False)

### Helper Functions

def load_feat(feat_wildcard,feat_path_list):
    feature_dict = { "mean" : Means, "mean-activity": Means, "spot-activity": Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup, "cofluctuation":Cofluctuation }
    feature_class = feature_dict[feat_wildcard.split("_")[0]]

    feat_list = []
    for path in feat_path_list:
        feat_list.append(feature_class.load(path))

    return feat_list

def balance(feat_list):
    ''' Balance number of trials between conditions'''
    min_number_trials = np.min([feats.trials_n for feats in feat_list])

    return [feat.subsample(min_number_trials) for feat in feat_list]   

def shuffle(labels):
    ### Shuffle     #TODO shuffle better integrated
    def shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)

    return shuffle_along_axis(labels,axis=0)


def flatten(feat_list,label_list, timepoints = slice(None)):
    ''' Flatten feature of data point with corresponding labels; You can select specific timepoints of the feature before flattening'''
    data_flat = np.concatenate([feat.flatten(timepoints) for feat in feat_list])
    labels_flat = np.concatenate([np.full((len(feat_list[i].flatten(timepoints))), label_list[i])
                             for i in range(len(feat_list))])


    return data_flat,labels_flat

def confusion_matrix(x,y_true,DecoderObject, label_order):
    y_pred = DecoderObject.predict(x)
    return metrics.confusion_matrix(y_true,y_pred,labels=label_order)


'''
#Labels in input beinhalten
def decode_from_feature(feat_wildcard,feat_path_list,labels,decoder,reps=5,outputs=None,balance=False,time_resolved=False):
    #Define Feature types to call corresponding loading function
    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup, "cofluctuation":Cofluctuation }
    feature_class = feature_dict[feat_wildcard.split("_")[0]]

    #Load features from each condition
    cond_feats = []
    for path in feat_path_list:
        cond_feats.append(feature_class.load(path))

    #Balance Conditions
    balance = True
    if balance:
        min_number_trials = cond_feats[0].trials_n
        for i in range(len(cond_feats)):
            print(labels[i])
            print(cond_feats[0].trials_n)
            min_number_trials = np.min([min_number_trials,cond_feats[i].trials_n])


        for i in range(len(cond_feats)):
            cond_feats[i].subsample(min_number_trials)


    #Flatten
    data_flat = np.concatenate([feat.flatten() for feat in cond_feats])
    labels_flat = np.concatenate([np.full((len(cond_feats[i].flatten())), labels[i])
                             for i in range(len(cond_feats))])

    decode(data_flat,labels_flat,decoder,reps,outputs,balance,time_resolved)

    if time_resolved:
        print(time_resolved)
        if cond_feats[0].is_time_resolved:
            for i in range(cond_feats[0].timepoints):
                    print("Timepoint")
                    print(i)
                    data_flat = np.concatenate([feat.flatten(timepoints=i) for feat in cond_feats])
                    labels_flat = np.concatenate([np.full((len(cond_feats[i].flatten(timepoints=i))), labels[i])
                                for i in range(len(cond_feats))])

                    decode(data_flat,labels_flat,decoder,reps,outputs,balance,time_resolved)    
'''

@ignore_warnings(category=ConvergenceWarning)
def decode(data, labels, decoder, reps, label_order=None,cores=1):
    ### Split
    cv = StratifiedShuffleSplit(reps, test_size=0.2, random_state=420)

    ### Scale
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    cv_split = cv.split(data, labels)
    perf = np.zeros((reps),dtype=float)
    trained_decoders = np.zeros((reps),dtype=object) 
    confusion = np.zeros((reps,len(label_order),len(label_order)),dtype=float)

    #Select Decoder
    if isinstance(decoder,str):
        decoders = {"MLR":MLR,
                "MLRshuffle":MLR, #TODO move to parameters
                "1NN":NN,
                "LDA":LDA,
                "RF":RF,
                "RFshuffle":RF}

        model = decoders[decoder](cores)

    ### Train & Eval
    try:
        for i, (train_index, test_index) in enumerate(cv_split):
            if isinstance(decoder,str):
                #If decoder is reference to a decoder, it wasn't trained yet
                model.fit(data[train_index,:],labels[train_index])
            else:
                #Otherwise its assumed to be an array of already trained decoders
                model = decoder[i]
                
            perf[i] = model.score(data[test_index,:],labels[test_index])
            confusion[i,:,:] = confusion_matrix(data[test_index,:],labels[test_index],model,label_order)
            trained_decoders[i] = model
    except Exception as Err:
        print("Error during training and testing")
        print(Err)

    
    return perf, confusion, trained_decoders


    
'''
def decode(data,labels,decoder,reps=5,outputs=None,balance=False,time_resolved=False):

    ### Shuffle     #TODO shuffle better integrated
    def shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a,idx,axis=axis)
    if "shuffle" in decoder:
        print(labels)
        labels = shuffle_along_axis(labels,axis=0)
        print(labels)


    ### Split
    cv = StratifiedShuffleSplit(reps, test_size=0.2, random_state=420)

    ### Scale
    scaler = preprocessing.StandardScaler().fit( data )
    data = scaler.transform(data)
    cv_split = cv.split(data, labels)
    perf = np.zeros((reps))
    trained_decoders = []
    
    #Select Decoder
    decoders = {"MLR":MLR,
                "MLRshuffle":MLR,
                "1NN":NN,
                "LDA":LDA,
                "RF":RF}
    decoder = decoders[decoder]()

    ### Train & Eval
    try:
        for i, (train_index, test_index) in enumerate(cv_split):
            decoder.fit(data[train_index,:],labels[train_index])
            perf[i] = decoder.score(data[test_index,:],labels[test_index])
            trained_decoders.append(decoder)
    except Exception as Err:
        print("Error during training and testing")
        print(Err)


    print(perf)

    #Save outputs
    #save_h5(perf, snakemake.output[1]) can't load with corresponding load function
    with open(outputs[1], 'wb') as f:
        pickle.dump(perf, f)

    with open(outputs[0], 'wb') as f:
        pickle.dump(decoders, f)
'''


