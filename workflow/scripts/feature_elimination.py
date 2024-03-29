from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))


from ci_lib.utils import snakemake_tools
from ci_lib.features import Features, Feature_Type, from_string as feat_from_string
from ci_lib.utils.logging import start_log
from ci_lib.feature_selection import RFE_pipeline, construct_rfe_graph, rec_feature_elimination

# redirect std_out to log file
logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    snakemake_tools.save_conf(snakemake, sections=["parcellations","selected_trials","conditions","features","decoders"],
                              params=['conds','reps'])
    timer_start = snakemake_tools.start_timer()

    ### Load features and labels for all conditions
    class_labels = snakemake.params['conds']
    feature = snakemake.wildcards["feature"]
    feature_class = feat_from_string(snakemake.wildcards["feature"].split("_")[0])

    class_feats = []
    for path in snakemake.input["feats"]:
        class_feats.append(feature_class.load(path))

    select_feats_n = snakemake.wildcards["rfe_n"]
    repetitions = snakemake.params['reps']


    #Select feats with RFE
    selected_feats, perf, decoders = rec_feature_elimination(select_feats_n, class_feats, class_labels, repetitions)


    #Save outputs
    snakemake_tools.save(snakemake.output["perf"],perf)
    snakemake_tools.save(snakemake.output["best_feats"],selected_feats)
    snakemake_tools.save(snakemake.output["model"],decoders)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
