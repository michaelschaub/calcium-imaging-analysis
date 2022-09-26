###   Data processing   ###


rule pipeline_entry:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        #To make sure that files are present, unfortunatly gets flattened -> losing information which dates belong to which subject
        _		= [[ f"resources/experiment/{subject_id}/{date}/task_data/"
                      for date in dates] for subject_id,dates in subjects.items()],
        Vc		= [ f"resources/experiment/{subject_id}/{date}/SVD_data/Vc.mat"
                      for subject_id,dates in subjects.items() for date in dates],
        trans_params	= [ f"resources/experiment/{subject_id}/{date}/SVD_data/opts.mat"
                            for subject_id,dates in subjects.items() for date in dates],
    output:
        f"results/{{subject_dates}}/SVD/data.h5",
        config = f"results/{{subject_dates}}/SVD/conf.yaml",
    params:
        subject_dates_str = '_'.join(subject_dates),
        #maybe find a clean solution from flattened array,
        task_structured = {subject_id: [ f"resources/experiment/{subject_id}/{date}/task_data/"
                                         for date in dates] for subject_id,dates in subjects.items()} # so we are using this one and we can actually use a dict to make it even comfier
    wildcard_constraints:
        subject_dates	= r"GN[a-zA-Z\d_-]+",
    log:
        f"results/{{subject_dates}}/SVD/pipeline_entry.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,1000)
    script:
        "scripts/default_entry.py"

rule mSM_entry:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        sessions	= [ f"resources/experiment/{subject_id}/{date}/SpatialDisc_Session.mat"
                        for subject_id,dates in subjects.items() for date in dates],
        Vc		= [ f"resources/experiment/{subject_id}/{date}/Vc.mat"
                      for subject_id,dates in subjects.items() for date in dates],
        trans_params	= [ f"resources/experiment/{subject_id}/{date}/opts2.mat"
                            for subject_id,dates in subjects.items() for date in dates],
    output:
        f"results/{{subject_dates}}/SVD/data.h5",
        config = f"results/{{subject_dates}}/SVD/conf.yaml",
    params:
        subject_dates_str = '_'.join(subject_dates),
        sessions_structured = {subject_id: { date: f"resources/experiment/{subject_id}/{date}/SpatialDisc_Session.mat"
                                             for date in dates} for subject_id,dates in subjects.items()}
    wildcard_constraints:
        subject_dates	= r"mSM[a-zA-Z\d_-]+",
    log:
        f"results/{{subject_dates}}/SVD/pipeline_entry.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,1000)
    script:
        "scripts/mSM_entry.py"

def parcellation_input(wildcards):
    input = {
        "data"	: f"{{data_dir}}/SVD/data.h5",
        "config": f"{{data_dir}}/SVD/conf.yaml" }
    branch = parcellations[wildcards["parcellation"]]["branch"]
    input.update( config["paths"]["parcellations"][branch] )
    return input

rule parcellation:
    '''
    decomposes data into different parcellations
    '''
    input:
        unpack(parcellation_input)
    params:
        params = lambda wildcards: parcellations[wildcards["parcellation"]]
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
        "trial_conditions" : { wildcards["cond"] : trial_conditions[wildcards["cond"]]},
        "phase_conditions" : { wildcards["cond"] : phase_conditions[wildcards["cond"]]},
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
        data = f"{{data_dir}}/data.h5",
        config = f"{{data_dir}}/conf.yaml",
    output:
        f"{{data_dir}}/{{feature}}/features.h5",
        config = f"{{data_dir}}/{{feature}}/conf.yaml",
    params:
        params = lambda wildcards: features[wildcards["feature"]]
    log:
        f"{{data_dir}}/{{feature}}/feature_calculation.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,2000)
    script:
        "scripts/feature.py"

rule feature_elimination:
    input:
        feats = [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in trial_conditions],
    output:
        best_feats	= f"{{data_dir}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/best_feats.{export_type}",
        model		= f"{{data_dir}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/decoder_model.pkl",
        perf		= f"{{data_dir}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/decoder_perf.{export_type}",
        config		= f"{{data_dir}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/conf.yaml",
    params:
        conds = list(trial_conditions),
        reps = rfe_reps
    log:
        f"{{data_dir}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/feature_calculation.log"
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "scripts/feature_elimination.py"

rule decoding:
    input:
        [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in trial_conditions],
    output:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{{decoder}}/decoder_model.pkl",
        f"{{data_dir}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{{decoder}}/decoder_perf.pkl",
        config = f"{{data_dir}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{{decoder}}/conf.yaml",
    params:
        conds = list(trial_conditions),
        params = lambda wildcards: decoders[wildcards["decoder"]]
    log:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{{decoder}}/decoding.log",
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "scripts/decoding.py"