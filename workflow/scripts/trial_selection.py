# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

from ci_lib.data import DecompData

### Setup
# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    #snakemake_tools.check_conf(snakemake, sections=["parcellations"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials"])
    timer_start = snakemake_tools.start_timer()

    params = snakemake.params[0]
    logger.debug(f"{params=}")
    if params['alias']:
        input_path = Path(snakemake.input[0])
        output_path = Path(snakemake.output[0])
        base_path = input_path.parent.parent.parent.parent
        raise NotImplemented("Aliasing not yet implemented")

        snakemake_tools.stop_timer(timer_start, logger=logger)
        sys.exit(0)

    ### Load
    selection_id = params['branch']
    logger.debug(f"{selection_id=}")
    data = DecompData.load(snakemake.input[0])

    ### Processing
    logger.debug(f"{data._df=}")
    logger.debug(f"{data._df.columns=}")
    logger.info(f"Applied selection {selection_id}")

    ### Save

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
