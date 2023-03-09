import pandas as pd
import pickle

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #Load Dfs
    dfs = []
    for df_path in snakemake.input:
        dfs.append(pd.read_pickle(df_path))
        logger.debug(f"Loaded frame from {df_path}.")

    #Concat
    dfs = pd.concat(dfs)
    logger.debug(f"Concated frames.")
    #dfs["dataset"] = snakemake.params[dataset_name]

    #Save Dfs
    dfs.to_pickle(snakemake.output[0])

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
