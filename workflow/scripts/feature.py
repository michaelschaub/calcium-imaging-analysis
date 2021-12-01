# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"calciumimagingtools").absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry","parcellation","prefilters","conditions"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","prefilters","conditions","feature_calculation"])
start = snakemake_tools.start_timer()

config = snakemake.config["rule_conf"]["feature_calculation"]
from data import DecompData
from features import Means, Raws, Covariances, AutoCovariances, Moup

feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "autocovariance" : AutoCovariances, "moup" :Moup }

param_dict = {
        "mean"              : (lambda p : {}),
        "raw"               : (lambda p : {}),
        "covariance"        : (lambda p : {}),
        "autocovariance"    : (lambda p : { "time_lag_range" : range(1,p["max_timelag"]+1) if "max_timelag" in p else p["timelags"] }),
        "moup"              : (lambda p : p)}

feature = snakemake.wildcards["feature"].split("_")[0]
params = snakemake.params["params"]
data = DecompData.load(snakemake.input[0])

feat = feature_dict[feature].create(data, max_comps=config["max_components"], **param_dict[feature](params))


feat.save(snakemake.output[0])

snakemake_tools.stop_timer(start, f"{snakemake.rule}")
