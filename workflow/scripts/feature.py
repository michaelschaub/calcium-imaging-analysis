from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
from ci_lib.features import Means, Raws, Covariances, Correlations, AutoCovariances, AutoCorrelations, Moup, Cofluctuation
 
# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["parcellations","selected_trials","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features"])
    start = snakemake_tools.start_timer()

    feature_dict = { "mean" : Means, "mean-activity": Means, "spot-activity": Means, "raw" : Raws, "covariance" : Covariances, "correlation" : Correlations, "autocovariance" : AutoCovariances, "autocorrelation" : AutoCorrelations, "moup" :Moup, "cofluctuation":Cofluctuation }

    # Dictionary for converting parameters from workflow to parameters, that can be passed to feature creators
    param_dict = {
            "spot-activity" : (lambda p : {"window": 1}),
            "mean"              : (lambda p : {}),
            "mean-activity" : (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]), "stop":int(snakemake.config["phase"][p["phase"]]["stop"])}),
            "raw"               : (lambda p : {}),
            "covariance"        : (lambda p : {}),
            "correlation"        : (lambda p : {}),
            # convert parameter "max_timelag" to range up to that timelag, if "max_timelag" does not exist, pass "timelags" (iterable)
            "autocovariance"    : (lambda p : { "timelags" : range(1,p["max_timelag"]+1) if "max_timelag" in p else p["timelags"] }), #TODO do we actually need ranges of timelags for a single feat?
            "autocorrelation"    : (lambda p : { "timelags" : range(1,p["max_timelag"]+1) if "max_timelag" in p else p["timelags"] }),
            # no conversion needed for Moup
            "moup"              : (lambda p : {"timelag": p["timelags"]}),
            "cofluctuation": (lambda p : {})}

    feature = snakemake.params["params"]['branch']
    params = snakemake.params["params"]
    data = DecompData.load(snakemake.input[0])

    max_comps = params["max_components"] if "max_components" in params else None
    window = params["window"] if "window" in params else None #TODO move param_dict , this will break for all feature that don't have window parameter!!

    feat = feature_dict[feature].create(data, max_comps=max_comps, **param_dict[feature](params)) #TODO **param_dict doesn't pass anything for empty lambda expression, shouldn't it just pass everyhting + replacement options
    logger.debug(f"feature shape {feat.feature.shape}")

    feat.save(snakemake.output[0])
    feat.plot(snakemake.output["export_plot"])
    snakemake_tools.save(snakemake, snakemake.output["export_raw"], feat.feature)
    #snakemake_tools.save(snakemake, snakemake.output["export_plot"], feat.feature)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
