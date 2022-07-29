# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["entry","parcellation"])
    snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","trial_selection"])
    timer_start = snakemake_tools.start_timer()

    import shutil

    if snakemake.wildcards['trials'] == 'All':
        shutil.copyfile( snakemake.input[0], snakemake.output[0])
    else:
        raise ValueError(f"Filter f{snakemake.wildcards['trials']} not recognized")
    logger.info(f"Applied filter {snakemake.wildcards['trials']}")

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
