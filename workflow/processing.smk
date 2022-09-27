from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config

###   Data processing   ###

def parcellation_input(wildcards):
    input = {
        "data"	: f"{{data_dir}}/SVD/data.h5",
        "config": f"{{data_dir}}/SVD/conf.yaml" }
    branch = config["parcellations"][wildcards["parcellation"]]["branch"]
    input.update( config["parcellation_wildcard_matching"][branch] )
    return input

rule parcellation:
    '''
    decomposes data into different parcellations
    '''
    input:
        unpack(parcellation_input)
    params:
        params = lambda wildcards: config["parcellations"][wildcards["parcellation"]]
    output:
        f"{{data_dir}}/{{parcellation}}/data.h5",
        config = f"{{data_dir}}/{{parcellation}}/conf.yaml",
    wildcard_constraints:
        # exclude SVD as parcellation
        parcellation = "(?!SVD).+"
    log:
        f"{{data_dir}}/{{parcellation}}/parcellation.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "scripts/parcellation.py"

use rule parcellation as locaNMF with:
    threads:
        workflow.cores*0.75
    wildcard_constraints:
        parcellation = "LocaNMF"
    conda:
        "envs/locaNMF_environment.yaml"

rule trial_selection:
    '''
    can select trials through predefined filters
    '''
    input:
        data = f"{{data_dir}}/data.h5",
        config = f"{{data_dir}}/conf.yaml",
    output:
        f"{{data_dir}}/{{trials}}/data.h5",
        #report = report(f"{{data_dir}}/{{trials}}/conf.yaml"),
        config = f"{{data_dir}}/{{trials}}/conf.yaml",
    log:
        f"{{data_dir}}/{{trials}}/trial_selection.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "scripts/trial_selection.py"

def condition_params(wildcards):
    params = {
        "trial_conditions" : { wildcards["cond"] : config["trial_conditions"][wildcards["cond"]]},
        "phase_conditions" : { wildcards["cond"] : config["phase_conditions"][wildcards["cond"]]},
    }
    return params

rule condition:
    '''
    Filters trials into different configured conditions
    '''
    input:
        data = f"{{data_dir}}/data.h5",
        config = f"{{data_dir}}/conf.yaml",
    output:
        f"{{data_dir}}/Features/{{cond}}/data.h5",
        config = f"{{data_dir}}/Features/{{cond}}/conf.yaml",
    params:
        condition_params
    log:
        f"{{data_dir}}/Features/{{cond}}/conditionals.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,2000,1000)
    script:
        "scripts/conditional.py"

rule feature_calculation:
    input:
        data = f"{{data_dir}}/{{cond}}/data.h5",
        config = f"{{data_dir}}/{{cond}}/conf.yaml",
    output:
        f"{{data_dir}}/{{cond}}/{{feature}}/features.h5",
        export = report(
            f"{{data_dir}}/{{cond}}/{{feature}}/features.{config['export_type']}",
            caption="report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}"}),
        config = f"{{data_dir}}/{{cond}}/{{feature}}/conf.yaml",
    params:
        params = lambda wildcards: config["features"][wildcards["feature"]]
    log:
        f"{{data_dir}}/{{cond}}/{{feature}}/feature_calculation.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,2000)
    script:
        "scripts/feature.py"

rule feature_elimination:
    input:
        feats = [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in config['trial_conditions']],
    output:
        best_feats	= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/best_feats.{config['export_type']}",
        model		= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/decoder_model.pkl",
        perf		= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/decoder_perf.{config['export_type']}",
        config		= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/conf.yaml",
    params:
        conds = list(config["trial_conditions"]),
        reps = config['feature_selection']['reps']
    log:
        f"{{data_dir}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/feature_calculation.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "scripts/feature_elimination.py"

rule decoding:
    input:
        [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in config['trial_conditions']],
    output:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoder_model.pkl",
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoder_perf.pkl",
        config = f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    params:
        conds = list(config['trial_conditions']),
        params = lambda wildcards: config["decoders"][wildcards["decoder"]]
    log:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoding.log",
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "scripts/decoding.py"