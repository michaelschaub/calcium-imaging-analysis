import pandas as pd
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
from ci_lib.features import from_string as feat_from_string

# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.check_conf(snakemake, sections=["parcellations","selected_trials","conditions"])
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features"])
    start = snakemake_tools.start_timer()
    ####
    #TODO better solutions for synonyms
    # Dictionary for converting parameters from workflow to parameters, that can be passed to feature creators
    param_dict = {
            #Activity 
            "spot-activity" : (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"])if "phase" in p else None,"window": 1 if "window" not in p else p["window"]}),
            "mean-activity" : (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]), "stop":int(snakemake.config["phase"][p["phase"]]["stop"])}),
            "full-activity" : (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"])if "phase" in p else None,"window": 1 if "window" not in p else p["window"],"full":True}),

            #Functional Connectivity
            "cofluctuation": (lambda p : {}), 
            "dFC": (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"]) if "phase" in p else None, "mean":False}),
            "FC": (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"]) if "phase" in p else None,"mean":True}),
            "full-dFC": (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"]) if "phase" in p else None, "mean":False, "full":True}),

            "spot-activity-dFC":  (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"]) if "phase" in p else None, "mean":False, "include_dia":True}),
            "full-activity-dFC": (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"]) if "phase" in p else None, "mean":False, "full":True,"include_dia":True}),
            "mean-activity-FC": (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]) if "phase" in p else None, "stop":int(snakemake.config["phase"][p["phase"]]["stop"]) if "phase" in p else None,"mean":True, "include_dia":True}),

            #Effective Connectivity
            "moup": (lambda p : {"start":int(snakemake.config["phase"][p["phase"]]["start"]), "stop":int(snakemake.config["phase"][p["phase"]]["stop"]),"timelag": p["timelag"] if 'timelag' in p else 1}),
            
            #Legacy
            #"mean"              : (lambda p : {}), #TODO remove
            #"raw"               : (lambda p : {}), #TODO remove
            "covariance"        : (lambda p : {}),
            "correlation"        : (lambda p : {}),
            "autocovariance"    : (lambda p : { "timelag" :  p['timelag'] if 'timelag' in p else 1 }),
            "autocorrelation"    : (lambda p : { "timelag" :  p['timelag'] if 'timelag' in p else 1 }),
            }

    feature = snakemake.params["params"]['branch']
    params = snakemake.params["params"]
    max_comps = params["max_components"] if "max_components" in params else None
    window = params["window"] if "window" in params else None #TODO move param_dict , this will break for all feature that don't have window parameter!!

    data = DecompData.load(snakemake.input[0])
    feat = feat_from_string(feature).create(data, max_comps=max_comps, **param_dict[feature](params)) #TODO **param_dict doesn't pass anything for empty lambda expression, shouldn't it just pass everyhting + replacement options
    logger.debug(f"feature condition {feat.frame['condition']}")
    logger.debug(f"feature shape {feat.feature.shape}")
    logger.debug(f"feature ids {feat.frame['trial_id']}")

    feat.frame['feature'] = feature
    feat.frame['feature_params'] = pd.Series({i: params for i in feat.frame.index})

    feat.save(snakemake.output[0])
    feat.plot(snakemake.output["export_plot"])
    snakemake_tools.save(snakemake.output["export_raw"], feat.feature)
    #snakemake_tools.save(snakemake.output["export_plot"], feat.feature)

    snakemake_tools.stop_timer(start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
