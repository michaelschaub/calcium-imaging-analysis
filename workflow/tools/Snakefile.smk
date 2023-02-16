configfile: "config/config.yaml"

from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config

print(config)

subjects = config["branch_opts"]["subjects"]
subject_dates = [".".join([subject_id,date]) for subject_id,dates in subjects.items() for date in dates ]
generalize_from = config["branch_opts"]["generalize_from"]

if config["branch_opts"].get('include_individual_sessions', False):  # config["branch_opts"].get('combine_sessions', False): #TODO fail safe config loading with defaults
    session_runs = subject_dates + ['#'.join(subject_dates)] 
else:
    session_runs = ['#'.join(subject_dates)]  #["All"]

combine_sessions = config["branch_opts"]["combine_sessions"]

print(session_runs)

parcells_conf   = config["branch_opts"]["parcellations"]
parcells_static = config["static_params"]["parcellations"]
parcellations   = create_parameters( parcells_conf, parcells_static )

selected_trials = config["branch_opts"]["selected_trials"]

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

config["loading"] = {"subjects": subjects,
                    "combine_sessions":combine_sessions,
                    "subject_dates"	:subject_dates}

config["output"] = {"processed_dates" :  session_runs}

print('phases:', phase_conditions)

config["processing"] = {"combine_sessions":combine_sessions,
                        "aggr_conditions" : aggr_conditions,
                        "trial_conditions" : trial_conditions,
                        "phase_conditions": phase_conditions,
                        "group_conditions" : group_conditions,
                        "feature_selection": {"n": rfe_ns,
                                              "reps": rfe_reps},
                        "parcellations": parcellations,
                        "parcellation_wildcard_matching": config["paths"]["parcellations"], #TODO what is this?

                        "features":features,
                        "decoders":decoders,
                        "session_runs":session_runs}


#For annotating plots
config["plotting"] =   {"plot_subject_labels": {f"{subject_id}(#{len(dates)})" for subject_id,dates in config["branch_opts"]["subjects"].items()},
                        "parcels_n" : config['branch_opts']['plotting']['plot_parcels']['n'],
                        "decoders" : decoders,
                        "subject_dates": subject_dates,
                        "aggr_conditions" : aggr_conditions,
                        "features": features,
                        "parcellations" :parcellations,
                        "default_conditions":default_conditions,
                        "generalize_from":generalize_from}

#config["generic"] = {"loglevel": config["loglevel"],
#                    "export_type": export_type,
#                    "limit_memory": 1,
#                    "branch_opts":config[branch_opts"}


#plot_feature_labels = [f"{feat_name} t={options['timelags']}" if 'timelags' in options else feat_name for feat_name,options in feature_conf.items()]

wildcard_constraints:
    subject_dates	= r"[a-zA-Z\d_.#-]+",
    parcellation  = branch_match(config["branch_opts"]["parcellations"].keys()),
    trials        = branch_match(config["branch_opts"]["selected_trials"].keys()),
    cond          = branch_match(config["branch_opts"]["conditions"].keys()),
    feature       = branch_match(config["branch_opts"]["features"].keys()),
    decoder       = branch_match(config["branch_opts"]["decoders"].keys()),

TRIALS_DIR = r"results/{session_runs}/{parcellation}/{trials}"
