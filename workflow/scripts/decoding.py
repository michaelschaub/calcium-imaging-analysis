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
import pandas as pd

#Multithreading
from threading import Thread

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools

from ci_lib.utils.logging import start_log
from ci_lib.features import Features, from_string as feat_from_string
from ci_lib.decoding import load_feat, balance, flatten, shuffle, decode

from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning) #TODO env vars don't work with multithreading

threads = []
next_thread = 0
thread_semaphore = False

def start_thread():
    global thread_semaphore, next_thread
    while thread_semaphore:
        pass
    thread_semaphore = True
    if(next_thread<len(threads)):
        threads[next_thread].start()
        next_thread += 1
    else:
        print("No more threads to start, waiting for other threads to finish")
    thread_semaphore = False

def decoding_iteration(t,feat_list,label_list,decoder,reps,perf_list,model_list,perf_matrix,accumulate=False,shuffling=False):
    #logger.info(f"Thread {t} started")

    if accumulate:
        t = range(t)

    #Flatten feature and labels from all conditions and concat
    feats_t, labels_t = flatten(feat_list,label_list,t)
    
    if shuffling:
        #Shuffle all labels as sanity check
        labels_t = shuffle(labels_t)

    #Decode
    perf_t, confusion_t, model_t = decode(feats_t, labels_t,decoder,reps,label_order= label_list)
    perf_list[t,:] = perf_t
    model_list[t]= model_t

    #Test on other timepoints
    for t2 in t_range:
        feats_t2, labels_t2 = flatten(feat_list,label_list,t2)
        perf_matrix[t,t2,:], confusion_t_not_used, _ = decode(feats_t2, labels_t2,model_t,reps,label_order= label_list)

    logger.info(f"Timepoint {t} finished")
    start_thread()




#Setup
# redirect std_out to log file
logger = start_log(snakemake)
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
    balancing = True #True #snakemake.params["params"]['balance']    ### Balance cond feats
    across_time = True
    accumulate = False

    #Load feature for all conditions
    feat_list = load_feat(snakemake.wildcards["feature"],snakemake.input)
    #print(snakemake.input)


    if balancing:
        #Balances the number of trials for each condition
        feat_list = balance(feat_list)
    
    #print(feat_list[0])
    #print(feat_list[0].feature)
    if feat_list[0].timepoints is None or "full" in snakemake.wildcards["feature"]: #TODO full matching just a workaround, save new full property of feature class 
        #1 Iteration for full feature (Timepoint = None)
        t_range = [None]
    else:
        #t Iterations for every timepoint t
        t_range = range(feat_list[0].timepoints)

        logger.info(f"{feat_list[0].feature.shape=}")
        logger.info(f"{feat_list[0].timepoints=}")

    logger.info(f"{t_range}")
    #Decoding results
    n_timepoints = len(list(t_range))
    n_classes = len(feat_list)

    perf_list = np.zeros((n_timepoints,reps))
    model_list = np.zeros((n_timepoints,reps),dtype=object)
    perf_matrix = np.zeros((n_timepoints,n_timepoints,reps))
    conf_matrix = np.zeros((n_timepoints,reps,n_classes,n_classes))
    norm_conf_matrix = np.zeros((n_timepoints,reps,n_classes,n_classes))

    #Performance stats
    iterations = np.zeros((n_timepoints,reps))
    t1_time = np.zeros((n_timepoints))
    t2_time = np.zeros((n_timepoints))

    #Multithreading
    cores = snakemake.threads
    multithreading = False #False# T

    if multithreading:
        threads_n = snakemake.threads -1 
        threads = [Thread(target=decoding_iteration, args=(t,feat_list,label_list,decoder,reps,perf_list,model_list,perf_matrix,accumulate,shuffling)) for t in t_range]

        #start intitial_threads, each thread will start a new one after it is finished
        next_thread = threads_n
        for t in t_range[:threads_n]:
            threads[t].start()

        #Busy waiting until all threads started
        while next_thread<len(threads):
            pass

        #Wait for all threads to finish
        for thread in threads:
            thread.join()
            print(f"Thread {thread} finished")                  
    
    else:
        #Decode all timepoints (without multithreading)
        for t in t_range:

            if accumulate:
                t = range(t)

            #Flatten feature and labels from all conditions and concat
            feats_t, labels_t = flatten(feat_list,label_list,t)
            
            if shuffling:
                #Shuffle all labels as sanity check
                labels_t = shuffle(labels_t)

            t1_train_testing = snakemake_tools.start_timer()

            #Decode
            perf_t, confusion_t, norm_confusion_t, model_t = decode(feats_t, labels_t,decoder,reps,label_order= label_list,cores=cores,logger=logger)
            perf_list[t,:] = perf_t
            conf_matrix[t,:,:,:] = confusion_t
            norm_conf_matrix[t,:,:,:] = norm_confusion_t
            model_list[t]= model_t

            #Track stats
            #iterations[t,:] = [model[-1].n_iter_[-1] for model in model_t]
            t1_time[t] = snakemake_tools.stop_timer(t1_train_testing,silent=True)
            logger.info(f"Timepoint {t} decoded {len(feats_t)} trials with average accuracy {np.mean(perf_t)}")

            #Test on other timepoints
            if across_time:
                t2_testing = snakemake_tools.start_timer()
                for t2 in t_range:
                    feats_t2, labels_t2 = flatten(feat_list,label_list,t2)
                    perf_matrix[t,t2,:], _ , _ , _ = decode(feats_t2, labels_t2,model_t,reps,label_order= label_list,cores=cores,logger=logger) #TODo could be optimizied (run only once for each t2 on all t1)

                t2_time[t] = snakemake_tools.stop_timer(t2_testing,silent=True)

    logger.info(f"Finished {n_timepoints} timepoints with {reps} repetitions")
    logger.info(f"Training & Testing each timepoints on average: {np.mean(t1_time)} s")
    logger.info(f"Testing on others timepoints on average: {np.mean(t2_time)} s")


    # Construct wide dataframe
    #add , "SVD_space": snakemake.wildcards[""] later when data_dict is reverted
    accuracy_dict = {
            "decomposition_space" : feat_list[0].frame["decomposition_space"].iloc[0],
            "parcellation"        : feat_list[0].frame["parcellation"].iloc[0],
            "dataset_id"          : feat_list[0].frame["dataset_id"].iloc[0],
            "decoding_space"      : feat_list[0].frame["dataset_id"].iloc[0],
            "conditions"          : '.'.join(np.unique([ f.frame["condition"].iloc[0] for f in feat_list])),
            "feature"             : feat_list[0].frame["feature"].iloc[0],
            "feature_params"      : feat_list[0].frame["feature_params"].iloc[0],
            }
    accuracy_dict = [{"t": t, "run":r, "accuracy":run_perf, "decoder": snakemake.wildcards["decoder"], **accuracy_dict}
                     for t,timepoint_perfs in enumerate(perf_list) for r, run_perf in  enumerate(timepoint_perfs) ]
    accuracy_df =  pd.json_normalize(accuracy_dict)
    accuracy_df.to_pickle(snakemake.output["df"])


    #Save results
    # TODO replace with better file format
    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(perf_list, f)

    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(model_list, f)
    
    with open(snakemake.output[2], 'wb') as f:
        pickle.dump(perf_matrix, f)

    with open(snakemake.output["conf_m"], 'wb') as f:
        pickle.dump(conf_matrix, f)

    with open(snakemake.output["norm_conf_m"], 'wb') as f:
        pickle.dump(norm_conf_matrix, f)

    with open(snakemake.output["labels"], 'wb') as f:
        pickle.dump(label_list, f) 

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
