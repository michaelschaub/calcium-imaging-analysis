from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config, readable_dataset_id


def aggregate_input(wildcards):
    decode_space = wildcards["decode_space"]
    decomp_space = wildcards["decomp_space"]
    perf_file = "Decoding/decoder/{conditions}/{feature}/{decoder}/perf_df.pkl"
    if decomp_space == "dataset":
        decomp_base_dir = "results/data/{dataset_id}/{parcellation}"
        if decode_space == "dataset":
            inputs = f"{decomp_base_dir}/{{dataset_id}}/{perf_file}".format(**wildcards)
        elif decode_space == "subsets":
            inputs = [f"{decomp_base_dir}/{subset_id}/{perf_file}".format(**wildcards)
                for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                ]#if subset_id in config["session_runs"][wildcards["dataset_id"]]]
        elif decode_space == "sessions":
            inputs = [f"{decomp_base_dir}/{session_id}/{perf_file}".format(**wildcards)
                for session_id in config["dataset_sessions"][wildcards["dataset_id"]]
                ]#if session_id in config["session_runs"][wildcards["dataset_id"]]]
        else:
            raise ValueError
    elif decomp_space == "subsets":
        decomp_base_dir = "results/data/{subset_id}/{parcellation}"
        if decode_space == "subsets":
            inputs = [f"{decomp_base_dir}/{subset_id}/{perf_file}".format(**wildcards, subset_id=subset_id)
                    for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                    ]#if subset_id in config["session_runs"][wildcards["dataset_id"]]]
        elif decode_space == "sessions":
            inputs = [f"{decomp_base_dir}/{session_id}/{perf_file}".format(**wildcards, subset_id=subset_id)
                    for subset_id in config["dataset_groups"][wildcards["dataset_id"]]
                    for session_id in config["dataset_sessions"][subset_id]
                    ]#if session_id in config["session_runs"][wildcards["dataset_id"]]]
        else:
            raise ValueError
    elif decomp_space == "sessions":
        decomp_base_dir = "results/data/{session_id}/{parcellation}"
        if decode_space == "sessions":
            inputs = [f"{decomp_base_dir}/{session_id}/{perf_file}".format(**wildcards, session_id=session_id)
                    for session_id in config["dataset_sessions"][wildcards["dataset_id"]]
                    ]#if session_id in config["session_runs"][wildcards["dataset_id"]]]
        else:
            raise ValueError
    else:
        raise ValueError
    return inputs

rule aggregate_perf:
    input:
        aggregate_input
    output:
        f"results/plots/{{dataset_id}}/{{decode_space}}_in_{{decomp_space}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
    params:
        decode_space   = '{decode_space}',
        decomp_space   = '{decomp_space}',
        aggregated_from = '{decode_space} in {decomp_space}'
    log:
        f"results/plots/{{dataset_id}}/{{decode_space}}_in_{{decomp_space}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/df_aggregation.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/aggregate_results.py"
