import numpy as np
import h5py
import warnings
from snakemake.logging import logger
import pandas as pd

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


    rng = np.random.default_rng()
    frames = 10**6
    n_trials = int((frames)/200)-2

    rand_dict = {'mouse_id': ['random']*n_trials,
            'date_time': 'task_data',
            'trial_id': range(n_trials),
            'modality':  rng.integers(0,3,size=n_trials),
            'target_side_left':   rng.integers(0,2,size=n_trials),
            'response_left':   rng.integers(0,2,size=n_trials),
            'auto_reward':   rng.integers(0,2,size=n_trials),
            'both_spouts':   rng.integers(0,2,size=n_trials),
            'n_distractors':   list(rng.integers(0,2,size=(n_trials,6))),
            'n_targets':  list(rng.integers(0,2,size=(n_trials,6))),
            'cues_left':  list(rng.integers(0,2,size=(n_trials,6))),
            'cues_right':  list(rng.integers(0,2,size=(n_trials,6))),
            'cues_left_vis':  list(rng.integers(0,2,size=(n_trials,6))),
            'cues_right_vis':  list(rng.integers(0,2,size=(n_trials,6))),
            'cues_left_tact':  list(rng.integers(0,2,size=(n_trials,6))),
            'cues_right_tact':  list(rng.integers(0,2,size=(n_trials,6))),
            'control_condition_id': 0}


    random_session = pd.DataFrame.from_dict(rand_dict)
    random_Vc = rng.random((frames,300))
    random_U = rng.random(((300, 540, 640)))
    trial_starts= np.arange(200,frames-200,200)

    print(n_trials)


    random = DecompData( random_session, random_Vc, random_U, trial_starts)
    random.save( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
