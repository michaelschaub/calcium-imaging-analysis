import numpy as np
import pandas as pd
import datetime

# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib.data import DecompData

def parse_datetime(date):
    try:
        return np.array(datetime.datetime.strptime(date, "%Y-%m-%d").date(), dtype='datetime64[D]')
    except ValueError:
        pass
    try:
        #TODO find better way for this or require a year in the session path
        return np.array(datetime.datetime.strptime(date, "%m-%d").date().replace(year=2021), dtype='datetime64[D]')
    except ValueError:
        pass
    try:
        return np.array(datetime.datetime.strptime(date, "%Y-%m-%d_%H-%M-%S"), dtype='datetime64[s]')
    except ValueError:
        pass
    raise NotImplemented(f"The format of {date=} is not implemented.")

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
    if params['is_dataset']:
        sessions = [ {'subject_id':subj, 'datetime':parse_datetime(date)} for subj, date in params['sessions']]
        logger.debug(f"{sessions=}")
        data = data.dataset_from( sessions, selection_id )
    else:
        selection_split = selection_id.split('-')
        subject_id = selection_split[0]
        date = '-'.join(selection_split[1:])
        sessions = [ {'subject_id':subject_id, 'datetime':parse_datetime(date)} ]
        logger.debug(f"session={sessions[0]}")
        data = data.dataset_from( sessions, selection_id )
    logger.info(f"Applied selection {selection_id}")
    logger.debug(f"{data._df=}")

    ### Save
    data.save(snakemake.output[0])

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
