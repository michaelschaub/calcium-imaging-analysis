from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData

# Setup
# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["parcellations", "selected_trials"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations", "selected_trials", "conditions"])
    timer_start = snakemake_tools.start_timer()

    # Load input
    data = DecompData.load(snakemake.input[0])

    data.conditions = snakemake.params[0]["trial_conditions"]

    # Processing
    for cond, file in zip(data.conditions.keys(), snakemake.output[:-1]):
        phase = snakemake.params[0]["phase_conditions"][cond]
        if phase is None:
            start = None
            stop = None
        else:
            start = phase["start"] if "start" in phase else None
            stop = phase["stop"] if "stop" in phase else None
     
        # Save output
        data.conditions[cond, :, start:stop].save(file)


    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
