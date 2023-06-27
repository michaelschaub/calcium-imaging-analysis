from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from ci_lib.utils.snakemake_tools import start_timer, stop_timer, save
from ci_lib.utils.logging import start_log
from ci_lib.data import DecompData

logger = start_log(snakemake)
try:
    timer_start = start_timer()

    data = DecompData.load(snakemake.input[0])

    logger.debug("data.trials_n=%s", data.trials_n)
    logger.debug("data.n_components=%s", data.n_components)
    logger.debug("data.n_xaxis=%s", data.n_xaxis)
    logger.debug("data.n_yaxis=%s", data.n_yaxis)
    logger.debug("data.t_max=%s", data.t_max)
    #assert (data.temporals == data[:,:data.temporals.shape[1]].temporals).all()

    save(snakemake.output['temporals'], data.temporals)
    save(snakemake.output['spatials'], data.spatials)

    stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
