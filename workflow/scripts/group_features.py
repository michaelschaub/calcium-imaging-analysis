from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
from ci_lib.features import from_string as FeatFromString
from ci_lib.decoding import balance

# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    #TODO add config check
    #snakemake_tools.check_conf(snakemake, sections=["parcellations","selected_trials","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features"])
    start = snakemake_tools.start_timer()

    feature = snakemake.params["params"]['branch']
    feature_class = FeatFromString(feature)

    feats = []
    for i in snakemake.input:
        feat = feature_class.load(i)
        logger.debug(i)
        logger.debug(feat.feature)
        if feat.trials_n>0:
            feats.append(feat)

    seed = snakemake.config['seed'] if 'seed' in snakemake.config else None
    feats = balance(feats, seed=seed)
    feat = feats[0]
    feat.concat(feats, overwrite=True) #TODO fails when no trials are present
    feat.frame['condition'] = cond

    #feat = feature_dict[feature].create(data, max_comps=max_comps, **param_dict[feature](params))
    logger.debug(f"feature shape {feat.feature.shape}")

    feat.save(snakemake.output[0])
    feat.plot(snakemake.output["export_plot"])
    snakemake_tools.save(snakemake, snakemake.output["export_raw"], feat.feature)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
