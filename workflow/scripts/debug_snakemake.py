# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

#Logging
from snakemake.logging import logger

from ci_lib.utils.logging import start_log
from ci_lib.utils import snakemake_tools

# redirect std_out to log file
logger = start_log(snakemake)
#snakemake_tools.save_conf(snakemake, sections=[])

def dict_from_NamedList(nlist):
    return {**nlist}

def list_from_NamedList(nlist):
    d = { v for _,v in nlist.items()  }
    return [ i for i in nlist if i not in d ]

logger.info("script_dir: {}".format(snakemake.scriptdir))
logger.info("input: {}".format(snakemake.input))
logger.info("output: {}".format(snakemake.output))
logger.info("log: {}".format(snakemake.log))
logger.info("param: {}".format(snakemake.params))
logger.info("params_list: {}".format(list_from_NamedList(snakemake.params)))
logger.info("params_dict: {}".format(dict_from_NamedList(snakemake.params)))
logger.info("config: {}".format(snakemake.config))
