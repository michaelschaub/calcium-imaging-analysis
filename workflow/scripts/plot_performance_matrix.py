import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import yaml

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()
    with open(snakemake.input["perf"], "rb") as f:
        perf_matrix = np.asarray(pickle.load(f))

    #print(perf_matrix)

    sns.set(rc={'figure.figsize':(9 , 7.5)})

    framerate=15
    fig = sns.heatmap(np.mean(perf_matrix,axis=2), cmap = 'viridis',square=True, vmin =0, vmax = 1,xticklabels=framerate, yticklabels=framerate)

    plt.xlabel("t (test)")
    plt.ylabel("t (train)")


    fig.figure.savefig( snakemake.output[0] )

    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(fig.figure, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
