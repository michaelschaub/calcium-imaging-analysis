from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config, readable_dataset_id


def aggregate_input(wildcards):
    decode_level = wildcards["decode_level"]
    decomp_level = wildcards["decomp_level"]
    perf_file = "Decoding/decoder/{conditions}/{feature}/{decoder}/perf_df.pkl"
    if decomp_level == "dataset":
        decomp_base_dir = "results/data/{dataset_id}/{parcellation}"
        if decode_level == "dataset":
            inputs = f"{decomp_base_dir}/{{dataset_id}}/{perf_file}".format(**wildcards)
        elif decode_level == "subsets":
            inputs = [f"{decomp_base_dir}/{subset_id}/{perf_file}".format(**wildcards)
                for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                ]#if subset_id in config["session_runs"][wildcards["dataset_id"]]]
        elif decode_level == "sessions":
            inputs = [f"{decomp_base_dir}/{session_id}/{perf_file}".format(**wildcards)
                for session_id in config["dataset_sessions"][wildcards["dataset_id"]]
                ]#if session_id in config["session_runs"][wildcards["dataset_id"]]]
        else:
            raise ValueError
    elif decomp_level == "subsets":
        decomp_base_dir = "results/data/{subset_id}/{parcellation}"
        if decode_level == "subsets":
            inputs = [f"{decomp_base_dir}/{subset_id}/{perf_file}".format(**wildcards, subset_id=subset_id)
                    for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                    ]#if subset_id in config["session_runs"][wildcards["dataset_id"]]]
        elif decode_level == "sessions":
            inputs = [f"{decomp_base_dir}/{session_id}/{perf_file}".format(**wildcards, subset_id=subset_id)
                    for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                    for session_id in config["dataset_sessions"][subset_id]
                    ]#if session_id in config["session_runs"][wildcards["dataset_id"]]]
        else:
            raise ValueError
    elif decomp_level == "sessions":
        decomp_base_dir = "results/data/{session_id}/{parcellation}"
        if decode_level == "sessions":
            inputs = [f"{decomp_base_dir}/{session_id}/{perf_file}".format(**wildcards, session_id=session_id)
                    for session_id in config["dataset_sessions"][wildcards["dataset_id"]]
                    ]#if session_id in config["session_runs"][wildcards["dataset_id"]]]
        else:
            raise ValueError
    else:
        raise ValueError
    return inputs

def get_subset_id(wildcards,input):
    #TODO do this the right way
    decode_level = wildcards["decode_level"]
    if decode_level == "subsets":
        subset_ids = [ subset_id
                    for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                    ]#if subset_id in config["session_runs"][wildcards["dataset_id"]]]
    elif decode_level == "sessions":
        # WARNING: terrible hack
        sessions = [ f.split('/')[5] for f in input ]
        subsets = { subset_id : config["dataset_sessions"][subset_id]
                    for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                    }#if subset_id in config["session_runs"][wildcards["dataset_id"]]}
        subset_ids = [ [ subset for subset, sess in subsets.items() if sess == session_id ]
                    for session_id in sessions ]
        for i,s in enumerate(subset_ids):
            if len(s) == 0:
                subset_ids[i] = wildcards["dataset_id"]
            elif len(s) == 1:
                subset_ids[i] = s[0]
            else:
                raise ValueError("Session '{sessions[i]}' found in multiple subsets!")

    else:
        subset_ids = None
    return subset_ids

rule aggregate_perf:
    input:
        aggregate_input
    output:
        f"results/plots/{{dataset_id}}/{{decode_level}}_in_{{decomp_level}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
    params:
        dataset_id     = '{dataset_id}',
        subset_id      = get_subset_id,
        decode_level   = '{decode_level}',
        decomp_level   = '{decomp_level}',
        aggregated_from = '{decode_level} in {decomp_level}'
    log:
        f"results/plots/{{dataset_id}}/{{decode_level}}_in_{{decomp_level}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/df_aggregation.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/aggregate_results.py"
