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

    DecompData.load(snakemake.input[0])

    #Plot activity

    fig.savefig(snakemake.output[0])

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
