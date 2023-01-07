from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import os
import numpy as np

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
from ci_lib.plotting import draw_neural_activity

### Setup
logger = start_log(snakemake) # redirect std_out to log file
try:
    snakemake_tools.check_conf(snakemake, sections=["parcellations"])
    timer_start = snakemake_tools.start_timer()

    ### Load
    data = DecompData.load(snakemake.input[0])

    ### Process (Plot)
    ### Save
    draw_neural_activity(frames=np.sum(data._spats, axis=0),path=snakemake.output['combined'],plt_title=snakemake.wildcards['parcellation'],subfig_titles="",overlay=True,outlined=True, logger=logger)

    os.mkdir(Path(snakemake.output['single']))
    for i in range(min(len(data._spats),snakemake.params['n'])):
        title = None if data._spat_labels is None else data._spat_labels[i]
        draw_neural_activity(frames=data._spats[i],
                            path=Path(snakemake.output['single'])/"parcel_{}.png".format(i if title is None else title),
                            plt_title=snakemake.wildcards['parcellation'], subfig_titles=title, overlay=True, logger=logger)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
