# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

#Logging
from snakemake.logging import logger

from ci_lib.utils import snakemake_tools

# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
#snakemake_tools.save_conf(snakemake, sections=[])

logger.info("snakemake:", dir(snakemake))
logger.info("script_dir", snakemake.scriptdir)
logger.info("input:", snakemake.input)
logger.info("output:", snakemake.output)
logger.info("log:", snakemake.log)
logger.info("param:", snakemake.params)
logger.info("config:", snakemake.config)
