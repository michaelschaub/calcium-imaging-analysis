configfile: "config/config.yaml"

from snakemake_tools import create_datasets, create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config

generalize_from = config["branch_opts"]["generalize_from"]

datasets = config["branch_opts"]["datasets"]
if not 'All' in datasets.keys():
    datasets['All'] = { 'group': list(datasets.keys()) }


group_datasets = datasets
datasets, dataset_aliases = create_datasets(datasets, config)


#print(f"{datasets=}")
#print(f"{dataset_aliases=}")

#TODO repair
if config["branch_opts"].get('include_individual_sessions', False):  # config["branch_opts"].get('combine_sessions', False): #TODO fail safe config loading with defaults
    session_runs = list(datasets.keys()) + [f"{subj}-{date}" for subj, date in datasets[dataset_aliases['All']]]
else:
    session_runs = list(datasets.keys()) 

#TODO make config setting for this
#session_runs.pop("All")  

def get_key_by_val(mydict,val):
    return list(mydict.keys())[list(mydict.values()).index(val)]

def readable_dataset_id(hash,aliases=None):
    if aliases is None:
        aliases = dataset_aliases
    try:
        return '#'.join([get_key_by_val(aliases ,hash), hash])
    except:
        return hash


readable_session_runs = [ readable_dataset_id(hash) for hash in datasets.keys()]
readable_to_hash = [{readable_id: hash_id}  for readable_id,hash_id in zip(readable_session_runs,datasets.keys())]



# TODO remove, only a hotfix
session_runs = readable_session_runs

datasets_shared_space = {}#{dataset_aliases[group_name] : [get_key_by_val(dataset_aliases,dataset) for dataset in group['group']] for group_name, group in group_datasets.items() if 'group' in group}


dataset_aliases = {readable_id: readable_dataset_id(hash) for readable_id,hash in dataset_aliases.items()}
print(f"{datasets=}")

#


#Maps hash of group to itself (shared space) or to the contained sessions (ind space)
aggr_shared = {group_name: group_name for group_name in datasets.keys()} 
aggr_ind = {group_name: ind_group_session for group_name, ind_group_session in datasets.items()}

print(f"{aggr_shared=}")
print(f"{aggr_ind=}")
#TODO remove?
combine_sessions = True

#print(f"{session_runs=}")

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

config["processing"] = {"combine_sessions":combine_sessions,
                        "aggr_conditions" : aggr_conditions,
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
                        "aggr_shared":aggr_shared,
                        "aggr_ind":aggr_ind,
                        "readable_session_runs":readable_session_runs,
                        "readable_to_hash" : readable_to_hash,
                        "datasets_shared_space": datasets_shared_space }

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
