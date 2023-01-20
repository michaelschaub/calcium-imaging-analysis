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

from ci_lib.utils.logging import start_log
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup, Cofluctuation
from ci_lib.decoding import load_feat, balance, flatten, shuffle, decode

from warnings import simplefilter





#Setup
# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:

    start = snakemake_tools.start_timer()

    #Load models
    with open(snakemake.input["models"], "rb") as f:
        models = pickle.load(f)
    logger.info([m.coef_ for m in models])

    #Load params from Snakemake
    label_list = snakemake.params['conds'] #TODO maybe needs reordering depending on classes in models
    reps = snakemake.params["params"]['reps']

    shuffling = ("shuffle" in snakemake.params["decoders"])
    balancing = True #True #snakemake.params["params"]['balance']    ### Balance cond feats

    #Load feature for all conditions
    feat_list = load_feat(snakemake.wildcards["feature"],snakemake.input["feat"])
    logger.info(feat_list)
    #print(snakemake.input)
    ##

    if balancing:
        #Balances the number of trials for each condition
        feat_list = balance(feat_list)
    logger.info(feat_list)
    if feat_list[0].timepoints is None or "full" in snakemake.wildcards["feature"]: #TODO full matching just a workaround, save new full property of feature class 
        #1 Iteration for full feature (Timepoint = None)
        t_range = [0]
    else:
        #t Iterations for every timepoint t
        t_range = range(feat_list[0].timepoints)
    logger.info(t_range)    
    #Decoding results
    n_timepoints = len(list(t_range))
    n_classes = len(feat_list)
    n_models = len(models)
    logger.info(n_timepoints)
    logger.info([m.coef_ for m in models])
    perf_list = np.zeros((n_timepoints,n_models,reps))
    logger.info(perf_list.shape)
    
    conf_matrix = np.zeros((n_timepoints,n_models,reps,n_classes,n_classes))
    norm_conf_matrix = np.zeros((n_timepoints,n_models,reps,n_classes,n_classes))

    #Decode all timepoints (without multithreading)
    for t in t_range:

        #Flatten feature and labels from all conditions and concat
        feats_t, labels_t = flatten(feat_list,label_list,t)
        
        if shuffling:
            #Shuffle all labels as sanity check
            labels_t = shuffle(labels_t)

        ###
        #Decode
        for m, model in enumerate(models):
            logger.info(m)
            logger.info(model)
            logger.info(model.coef_.shape)
            perf_t, confusion_t, norm_confusion_t, _ = decode(feats_t, labels_t,[model]*reps,reps,label_order= label_list,logger=logger) #TODO model[1] is hardcoded for mlr pipeline, other decoders are just model
            logger.info(perf_t.shape)
            logger.info(f"{t}{t_range}")
            perf_list[t,m,:] = perf_t
            conf_matrix[t,m,:,:,:] = confusion_t
            norm_conf_matrix[t,m,:,:,:] = norm_confusion_t

            #Track stats
            logger.info(f"Timepoint {t} decoded {len(feats_t)} trials with average accuracy {np.mean(perf_t)}")


    #Save results
    # TODO replace with better file format
    with open(snakemake.output["perf"], 'wb') as f:
        pickle.dump(perf_list, f)
        '''
        with open(snakemake.output["conf_m"], 'wb') as f:
            pickle.dump(conf_matrix, f)

        with open(snakemake.output["norm_conf_m"], 'wb') as f:
            pickle.dump(norm_conf_matrix, f)

        with open(snakemake.output["labels"], 'wb') as f:
            pickle.dump(label_list, f) 
        '''
    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
