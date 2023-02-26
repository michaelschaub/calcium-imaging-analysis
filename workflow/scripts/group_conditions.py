from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
from ci_lib.decoding import balance

# Setup
# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations", "selected_trials", "conditions"])
    timer_start = snakemake_tools.start_timer()

    data = [DecompData.load(i) for i in snakemake.input]

    seed = snakemake.config['seed'] if 'seed' in snakemake.config else None
    data = balance(data, seed=seed)

    data[0].concat(data[1:], overwrite=False) #TODO fails when no trials are present
    data[0].frame['condition'] = snakemake.wildcards['cond']
    data[0].save(snakemake.output[0])


    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
