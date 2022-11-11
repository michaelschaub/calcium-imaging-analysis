from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup, Cofluctuation

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    #TODO add config check
    #snakemake_tools.check_conf(snakemake, sections=["parcellations","selected_trials","conditions"])
    #snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features"])
    start = snakemake_tools.start_timer()

    feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup, "cofluctuation":Cofluctuation }
    feature_class = feature_dict[snakemake.wildcards["feature"].split("_")[0]]


    feats = []
    for i in snakemake.input:
        feat = feature_class.load(i)
        if feat.trials_n>0:
            feats.append(feat)

    feats[0].concat(feats, overwrite=True)

    #feat = feature_dict[feature].create(data, max_comps=max_comps, **param_dict[feature](params))
    logger.debug(f"feature shape {feats[0].feature.shape}")

    feats[0].save(snakemake.output[0])

    feats[0].plot(snakemake.output["export_plot"])
    snakemake_tools.save(snakemake, snakemake.output["export_raw"], feats[0].feature)
    #snakemake_tools.save(snakemake, snakemake.output["export_plot"], feat.feature)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
