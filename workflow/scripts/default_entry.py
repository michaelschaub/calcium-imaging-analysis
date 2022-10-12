import numpy as np
import h5py
import warnings
from snakemake.logging import logger

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.loading import load_task_data_as_pandas_df, alignment #import extract_session_data_and_save
from ci_lib.plotting import draw_neural_activity
from ci_lib import DecompData

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    files_Vc = snakemake.input["Vc"]
    task_files = snakemake.params["task_structured"] #snakemake.input["tasks"]
    trans_paths = snakemake.input["trans_params"]
    subject_dates_str = snakemake.params["subject_dates_str"]

    sessions = load_task_data_as_pandas_df.extract_session_data_and_save(
            root_paths=task_files, mouse_dates_str = subject_dates_str, reextract=True, logger=logger) #reextraction needs to be done for different set of dates otherwise session will have wrong dims
    logger.info("Loaded task data")


    if len(files_Vc) > 1:
        warnings.warn("Combining different dates may still be buggy!")

    trial_starts = []
    Vc = []
    U = []
    start = 0
    for file_Vc,trans_path in zip(files_Vc,trans_paths):
        f = h5py.File(file_Vc, 'r')

        #Aligns spatials for each date with respective trans_params
        alignend_U, align_plot = alignment.align_spatials_path(np.array(f["U"]),trans_path,plot_alignment_path=snakemake.output["align_plot"]) #TODO export plot for each subject/session
        print(alignend_U.shape)
        U.append(alignend_U)

        frameCnt = np.array(f['frameCnt'])
        Vc.append(np.array(f["Vc"]))
        #assert np.array_equal(U[-1], U[0], equal_nan=True), "Combining different dates with different Compositions is not yet supported"
        # multiple dates cant have same spatials currently as they are independently created

        trial_starts.append(np.cumsum(frameCnt[:-1, 1]) + start) #TODO append will be deprecated, use concat
        start += Vc[-1].shape[0]

    trial_starts = np.concatenate( trial_starts )
    #Vc = np.concatenate( Vc )


    #

    mean_U = np.mean(U,axis=0)

    spat_n, w , h = mean_U.shape

    mean_U = mean_U.reshape(spat_n,w * h)
    print(mean_U.shape)

    mean_U_inv = np.nan_to_num(np.linalg.pinv(np.nan_to_num(mean_U, nan=0.0)), nan=0.0)



    error = np.zeros((len(U)))
    print(mean_U_inv.shape)
    print(len(Vc))
    for i,V in enumerate(Vc):
        print(V.shape)
        print(V.dtype)
        U[i] = U[i].reshape(spat_n,w * h)

        V_transform = np.matmul(np.nan_to_num(U[i], nan=0.0), mean_U_inv)

        Vc[i] = np.matmul(Vc[i], V_transform)
        #V    spat_n**

        #for t, frame in enumerate(V):

        #    correct_frame = Vc[i][t]

        #    Vc[i][t] = np.matmul(frame[np.newaxis,:] , V_transform)

            #Vc[i][t] = frame * U[i] * mean_U_inv
        #    error[i] += np.sum(np.absolute(Vc[i][t] - correct_frame))
        #print(error[i])
        print("after")
        print(Vc[i].shape)
        print(Vc[i].dtype)

        print(np.isnan(np.sum(Vc[i])))

        error[i] = np.linalg.norm(V_transform - np.eye(spat_n))



    U = mean_U.reshape(spat_n,w,h) # U[0]
    print(np.isnan(np.sum(U)))
    print(error)

    Vc = np.concatenate( Vc )




    svd = DecompData( sessions, Vc, U, trial_starts)
    svd.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
