# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"code").absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry","parcelation","prefilters","conditions"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcelation","prefilters","conditions","feature_calculation"])
start = snakemake_tools.start_timer()

config = snakemake.config["rule_conf"]["feature_calculation"]
from data import DecompData
from features import Means, Raws, Covariances, AutoCovariances, Moup

feature_dict = { "mean" : Means, "raw" : Raws, "covariance" : Covariances, "autocovariance" : AutoCovariances, "moup" :Moup }

feature = snakemake.wildcards["feature"]
data = DecompData.load(snakemake.input[0])

feat = feature_dict[feature].create(data[:,config["stim_start"]:config["stim_stop"]], max_comps=config["max_components"])


feat.save(snakemake.output[0])

snakemake_tools.stop_timer(start, f"{snakemake.rule}")
