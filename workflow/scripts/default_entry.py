import numpy as np
import h5py
import warnings
from snakemake.logging import logger

from sklearn.decomposition import TruncatedSVD

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib.plotting import draw_neural_activity
from ci_lib import DecompData

import pandas as pd
import seaborn as sns

# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    files_Vc = snakemake.input["Vc"]

    task_files = snakemake.params[0]['task_structured'] #snakemake.input["tasks"] #
    trans_paths = snakemake.input["trans_params"]

    


    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(
            root_paths=task_files, reextract=True, logger=logger) #reextraction needs to be done for different set of dates otherwise session will have wrong dims

    
    ###
    
    #ax =.plot.scatter(0,1)
    #ax.figure.savefig(snakemake.output["stim_side"])
    switch  = []
    same = []

    for i,trial in enumerate((sessions['target_side_left'])):
        if i>0:
            if prev_trial == trial:
                same.append(trial)
            else:
                switch.append(trial)
        prev_trial = trial
    same = np.asarray(same)
    switch = np.asarray(switch)

    left_switch = len(switch[switch==1])
    right_switch = len(switch[switch==0])
    right_same = len(same[same==0])
    left_same = len(same[same==1])

    sns.set()
    ax = sns.barplot(x=["left_switch","right_switch","left_same","right_same"],y=[left_switch,right_switch,left_same,right_same])


        #ax = sns.heatmap( pd.DataFrame(sessions['target_side_left'][:100]))
    ax.figure.savefig(snakemake.output["stim_side"])

    #ax = sns.heatmap( pd.DataFrame(sessions['target_side_left'][:100]))
    #ax.figure.savefig(snakemake.output["stim_side"])


    logger.info("Loaded task data")


    #if len(files_Vc) > 1:
    #    warnings.warn("Combining different dates may still be buggy!")

    trial_starts = []
    Vc = []
    U = []
    start = 0


    for file_Vc,trans_path in zip(files_Vc,trans_paths):
        f = h5py.File(file_Vc, 'r')

        #Aligns spatials for each date with respective trans_params
        alignend_U, align_plot = alignment.align_spatials_path(np.array(f["U"]),trans_path,plot_alignment_path=snakemake.output["align_plot"])
        U.append(alignend_U)

        frameCnt = np.array(f['frameCnt'])
        Vc.append(np.array(f["Vc"]))
        
        frames, n_components = Vc[-1].shape
        _, width, height = U[-1].shape
        logger.info(
            f"Dimensions: n_trials={len(frameCnt)-2}, average frames per trial={frames/(len(frameCnt)-2)}, n_components={n_components}, width={width}, height={height}")
        assert np.array_equal(U[-1].shape, U[0].shape, equal_nan=True), "Combining dates with different resolutions or number of components is not yet supported"

        trial_starts.append(np.cumsum(frameCnt[:-1, 1]) + start)
        start += Vc[-1].shape[0]

    trial_starts = np.concatenate( trial_starts )

    if len(U)==1:
        U = U[0]
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        flat_U = np.nan_to_num(np.concatenate([u.reshape(n_components,width * height) for u in U]))
        
        svd.fit(flat_U)
        
        print(svd.components_.shape)
        #mean_U = np.mean(np.nan_to_num(U),axis=0)


        #spat_n, w , h = mean_U.shape

        mean_U = svd.components_ 
        # mean_U = mean_U.reshape(spat_n,w * h)

        mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)

        #approx_U_error = np.matmul(mean_U_inv, mean_U)

        error = np.zeros((len(U)))

        for i,V in enumerate(Vc):
            U[i] = U[i].reshape(n_components,width * height)

            V_transform = np.matmul(np.nan_to_num(U[i], nan=0.0), mean_U_inv)

            Vc[i] = np.matmul(Vc[i], V_transform)
            #V    spat_n**

            #for t, frame in enumerate(V):

            #    correct_frame = Vc[i][t]

            #    Vc[i][t] = np.matmul(frame[np.newaxis,:] , V_transform)

                #Vc[i][t] = frame * U[i] * mean_U_inv
            #    error[i] += np.sum(np.absolute(Vc[i][t] - correct_frame))
            #print(error[i])

            #error[i] = np.linalg.norm(Vc[i]) * np.linalg.norm(U[i]) * np.linalg.norm(np.eye(spat_n)-approx_U_error)
            
            
            '''error_tmp = np.zeros((h))
            abs_val = np.zeros((h))
            range_ws = np.arange(0,w*h,w)
            for h_i,w_i in enumerate(range_ws):
                print(h_i)
                eye_vector = np.zeros((w,w*h))
                eye_vector[:,w_i:w+w_i] = np.eye(w)
                error_tmp[h_i] = np.sum(np.square(np.matmul(np.matmul(Vc[i],np.nan_to_num(U[i][:,w_i:w_i+w])),(eye_vector - np.matmul(mean_U_inv[w_i:w_i+w,:],np.nan_to_num(mean_U[:,:],nan=0.0))))))
                
                print((np.matmul(Vc[i],np.nan_to_num(U[i][:,w_i:w_i+w]))).shape)
                print(np.matmul(Vc[i],np.nan_to_num(U[i][:,w_i:w_i+w])))
                abs_val[i] = np.sum(np.square(np.matmul(Vc[i],np.nan_to_num(U[i][:,w_i:w_i+w]))))
                print(error_tmp[i])
                print(abs_val[i])

            error[i] = np.sum(error_tmp) / np.sum(abs_val)'''
            


        # for h_i,w_i in enumerate(range_ws):
        #     print(h_i)
        #     eye_vector = np.zeros((w*h,w))
        #     eye_vector[w_i:w+w_i,:] = np.eye(w)
        #     error_tmp[h_i] = np.sum(np.square((eye_vector - np.matmul(mean_U_inv[:,:],np.nan_to_num(mean_U[:,w_i:w_i+w],nan=0.0)))))
        #     print(error_tmp[i])


        
        '''print(f"Error for {snakemake.wildcards['subject_dates']}")
        print(error)
        print(error_tmp)'''
        

        U = mean_U.reshape(n_components,width,height) # U[0]
        
        
    #####
    

    Vc = np.concatenate( Vc )

    svd = DecompData( sessions, Vc, U, trial_starts)
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
