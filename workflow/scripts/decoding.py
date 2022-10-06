import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.discriminant_analysis as skda
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
import sklearn.ensemble as skens
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.features import Features, Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup

#Setup
# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features","decoders"],
                                            params=['conds','params'])
    start = snakemake_tools.start_timer()

    ### Load feature for all conditions
    cond_str = snakemake.params['conds']
    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup }
    feature_class = feature_dict[snakemake.wildcards["feature"].split("_")[0]]

    cond_feats = []
    for path in snakemake.input:
        cond_feats.append(feature_class.load(path))


    ### Select decoder
    def MLR():
        return skppl.make_pipeline(skppc.StandardScaler(),
                                   skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial',
                                                             solver='lbfgs', max_iter=500))

    def NN():
        return sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

    def LDA():
        return skda.LinearDiscriminantAnalysis(n_components=None, solver='eigen', shrinkage='auto')

    def RF():
        return skens.RandomForestClassifier(n_estimators=100, bootstrap=False)

    decoders = {"MLR":MLR,
                "1NN":NN,
                "LDA":LDA,
                "RF":RF}

    decoder = decoders[snakemake.params['params']['branch']]()

    ### Split
    rep = snakemake.params["params"]['reps']
    cv = StratifiedShuffleSplit(rep, test_size=0.2, random_state=420)

    data = np.concatenate([feat.flatten() for feat in cond_feats])
    labels = np.concatenate([np.full((len(cond_feats[i].flatten())), cond_str[i])
                             for i in range(len(cond_feats))])

    ### Scale
    scaler = preprocessing.StandardScaler().fit( data )
    data = scaler.transform(data)
    cv_split = cv.split(data, labels)
    perf = np.zeros((rep))
    decoders = []

    ### Train & Eval
    try:
        for i, (train_index, test_index) in enumerate(cv_split):
            decoder.fit(data[train_index,:],labels[train_index])
            perf[i] = decoder.score(data[test_index,:],labels[test_index])
            decoders.append(decoder)
    except:
        print("Error during training and testing")


    #Save outputs
    #save_h5(perf, snakemake.output[1]) can't load with corresponding load function
    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(perf, f)

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(decoders, f)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
