import pandas as pd
import pickle

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #Load Dfs
    dfs = []
    for df_path in snakemake.input:
        dfs.append(pd.read_pickle(df_path))

    #Concat
    dfs = pd.concat(dfs)
    dfs["dataset"] = snakemake.params[dataset_name]

    #Save Dfs
    dfs.to_pickle(snakemake.output[0])

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
