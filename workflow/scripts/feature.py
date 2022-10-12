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

    feature = snakemake.params["params"]['branch']
    params = snakemake.params["params"]
    max_comps = params["max_components"] if "max_components" in params else None
    params.pop("branch",None)
    params.pop("max_components",None)

    data = DecompData.load(snakemake.input[0])
    feat = feat_from_string(feature).create(data, max_comps=max_comps, **params)
    logger.debug(f"feature shape {feat.feature.shape}")


    feat.save(snakemake.output[0])

    snakemake_tools.save(snakemake, snakemake.output["export_raw"], feat.feature)

    #snakemake_tools.save(snakemake, snakemake.output["export_plot"], feat.feature)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
