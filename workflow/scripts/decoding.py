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
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup, Cofluctuation
from ci_lib.decoding import load_feat, balance, flatten, shuffle, decode

#Setup
# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features","decoders"],
                                            params=['conds','params'])
    start = snakemake_tools.start_timer()

    #Load params from Snakemake
    label_list = snakemake.params['conds']
    reps = snakemake.params["params"]['reps']
    decoder = snakemake.params['params']['branch']  #TODO why not from wildcard?
    shuffling = ("shuffle" in decoder)
    balancing = True #snakemake.params["params"]['balance']    ### Balance cond feats

    #Load feature for all conditions
    feat_list = load_feat(snakemake.wildcards["feature"],snakemake.input)
    
    if balancing:
        #Balances the number of trials for each condition
        feat_list = balance(feat_list)

    perf_list, model_list = [], []

    if feat_list[0].timepoints is None:
        #1 Iteration for full feature (Timepoint = None)
        t_range = [None]
    else:
        #t Iterations for every timepoint t
        t_range = range(feat_list[0].timepoints)

    for t in t_range:
        #Flatten feature and labels from all conditions and concat
        feats_t, labels_t = flatten(feat_list,label_list,t)
        
        if shuffling:
            #Shuffle all labels as sanity check
            labels_t = shuffle(labels_t)

        #Decode
        perf_t, model_t = decode(feats_t, labels_t,decoder,reps)
        perf_list.append(perf_t)
        model_list.append(model_t)

    #Save results
    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(perf_list, f)

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(model_list, f)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
