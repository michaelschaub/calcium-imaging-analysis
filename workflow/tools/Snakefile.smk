configfile: "config/config.yaml"

from snakemake_tools import create_datasets, create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config

generalize_from = config["branch_opts"]["generalize_from"]

datasets = config["branch_opts"]["datasets"]
if not 'All' in datasets.keys():
    datasets['All'] = { 'group': list(datasets.keys()) }


group_datasets = datasets
datasets, dataset_sessions, dataset_groups, dataset_aliases = create_datasets(datasets, config)


print(f"{datasets=}")
print(f"{dataset_sessions=}")
print(f"{dataset_groups=}")
print(f"{dataset_aliases=}")

unified_space = config['branch_opts'].get('unified_space', 'All')
individual_sessions = config["branch_opts"].get('include_individual_sessions', False)

if unified_space in ["Both", "Datasets"]:
    unification_groups = dataset_groups
    if unified_space != "Both":
        unification_groups.pop(dataset_aliases["All"])
else:
    all_id = dataset_aliases["All"]
    unification_groups = { all_id: dataset_groups[all_id] }

session_runs = {}
for set_id, sub_ids in unification_groups.items():
    sub_datasets = [set_id, *sub_ids]
    if individual_sessions:
        sub_datasets.extend(dataset_sessions[set_id])
    session_runs[set_id] = sub_datasets
print(f"{session_runs=}")

parcells_conf   = config["branch_opts"]["parcellations"]
parcells_static = config["static_params"]["parcellations"]
parcellations   = create_parameters( parcells_conf, parcells_static )

# Do still want to keep some sort of trial selection noit based on groups?
#TODO cartesian product with datasets
# selected_trials = config["branch_opts"]["selected_trials"]
# selected_trials = create_parameters( selected_trials, {})
'''Create a parameter dictionary for trial selection; the only parameters, appart from branch, is_dataset indicates, whether a single trial or a dataset is selected'''
trial_selection = { dataset_id: { "branch": dataset_id, "sessions": sessions, "is_dataset":True} for dataset_id, sessions in datasets.items() }
#print(f"{trial_selection=}")

selected_trials = { session: [session] for session in session_runs }
if dataset_aliases["All"] in selected_trials.keys():
    selected_trials[dataset_aliases["All"]] = list(datasets.keys())
#print(f"{selected_trials=}")

config["phase"] = config["phase_conditions"] #TODO check why phase_conditions is different from this

conditions    = config["branch_opts"]["conditions"]
aggr_conditions, trial_conditions, phase_conditions, group_conditions, default_conditions = create_conditions(conditions, config)

feature_conf    = config["branch_opts"]["features"]
feature_static  = config["static_params"]["features"]
features        = create_parameters( feature_conf, feature_static )

decoder_conf    = config["branch_opts"]["decoders"]
decoder_static  = config["static_params"]["decoders"]
decoders        = create_parameters( decoder_conf, decoder_static )

rfe_ns = config["branch_opts"]["rfe"]["select_features_n"]
rfe_reps = config["branch_opts"]["rfe"]["reps"]

rfe_ns = config["branch_opts"]["rfe"]["select_features_n"]
rfe_reps = config["branch_opts"]["rfe"]["reps"]

#Run id (based on hash of config)
run_id = hash_config(config)

config["loading"] = {"datasets"        : datasets,
                    "dataset_aliases" : dataset_aliases,
                    } #"subject_dates"  :subject_dates}

config["output"] = {"processed_dates" :  session_runs}

#print('phases:', phase_conditions)

config["processing"] = {"aggr_conditions" : aggr_conditions,
                        "trial_conditions" : trial_conditions,
                        "phase_conditions": phase_conditions,
                        "group_conditions" : group_conditions,
                        "feature_selection": {"n": rfe_ns,
                                              "reps": rfe_reps},
                        "parcellations": parcellations,
                        "parcellation_wildcard_matching": config["paths"]["parcellations"], #TODO what is this?
                        "trial_selection" : trial_selection,
                        "dataset_aliases" : dataset_aliases,
                        "features":features,
                        "decoders":decoders,
                        "session_runs":session_runs}


config["aggregation"] = {"dataset_groups" : dataset_groups,
                        "dataset_sessions" : dataset_sessions,
                        "session_runs":session_runs}

#For annotating plots
#TODO repair!
config["plotting"] =   {
                        #"plot_subject_labels": {f"{subject_id}(#{len(dates)})" for subject_id,dates in config["branch_opts"]["subjects"].items()},
                        "plot_subject_labels": {},
                        "parcels_n" : config['branch_opts']['plotting']['plot_parcels']['n'],
                        "decoders" : decoders,
                        #"subject_dates": subject_dates,
                        "aggr_conditions" : aggr_conditions,
                        "features": features,
                        "parcellations" :parcellations,
                        "default_conditions":default_conditions,
                        "generalize_from":generalize_from,
                        "dataset_aliases":dataset_aliases,
                        "session_runs":session_runs,
                        "datasets":datasets,
                        }

#config["generic"] = {"loglevel": config["loglevel"],
#                    "export_type": export_type,
#                    "limit_memory": 1,
#                    "branch_opts":config[branch_opts"}


#plot_feature_labels = [f"{feat_name} t={options['timelags']}" if 'timelags' in options else feat_name for feat_name,options in feature_conf.items()]

wildcard_constraints:
    subject_dates = r"[a-zA-Z\d_.#-]+",
    parcellation  = branch_match(config["branch_opts"]["parcellations"].keys()),
    #trials        = branch_match(config["branch_opts"][""].keys()),
    cond          = branch_match(config["branch_opts"]["conditions"].keys()),
    feature       = branch_match(config["branch_opts"]["features"].keys()),
    decoder       = branch_match(config["branch_opts"]["decoders"].keys()),

TRIALS_DIR = r"results/{session_runs}/{parcellation}/{trials}"
