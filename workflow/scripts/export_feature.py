from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.features import from_string as feat_from_string

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["parcellations","selected_trials","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features"])
    start = snakemake_tools.start_timer()


    feat = feat_from_string(snakemake.params["branch"]).load(path)
    snakemake_tools.save(snakemake.output[0], feat.feature, export_type=snakemake.wildcards['ext'])

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
