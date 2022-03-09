from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import numpy as np

from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.plotting import draw_neural_activity

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["entry","parcellation"])
    #snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","trial_selection","conditions"])
    timer_start = snakemake_tools.start_timer()

    data = DecompData.load(snakemake.input[0])

    draw_neural_activity(frames=np.sum(data._spats, axis=0),path=snakemake.output['combined'],plt_title=snakemake.wildcards['parcellation'],subfig_titles="")

    draw_neural_activity(frames=data._spats,path=snakemake.output['all'],plt_title=snakemake.wildcards['parcellation'],subfig_titles=data._spat_labels)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
