from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config
from wildcard_functions import subjects_from_wildcard

import re

###   Data processing   ###

def parcellation_input(wildcards):

    #if not bool(re.match(r"^(?!SVD)(.*)$",wildcards["parcellation"])):  #TODO why is snakemake regex broken? :( ,fro now evaulate the same freaking expression within input function and require non existing files to exclude this rule....
        #return {"data":f"good/luck/finding/this/non/existing/path"}

    input = {
        "data"	: f"{{data_dir}}/SVD/data.h5",
        "config": f"{{data_dir}}/SVD/conf.yaml" }
    branch = config["parcellations"][wildcards["parcellation"]]["branch"]
    input.update( config["parcellation_wildcard_matching"][branch] )
    return input

rule parcellate:
    '''
    decomposes data into different parcellations
    '''
    input:
        unpack(parcellation_input)
    output:
        f"{{data_dir}}/{{parcellation}}/data.h5",
        config = f"{{data_dir}}/{{parcellation}}/conf.yaml",
    params:
        params = lambda wildcards: config["parcellations"][wildcards["parcellation"]]
    wildcard_constraints:
        # exclude SVD as parcellation
        #TODO check if this really works
        parcellation = "(?!SVD$).+"
    log:
        f"{{data_dir}}/{{parcellation}}/parcellation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "../scripts/parcellation.py"

use rule parcellate as locaNMF with:
    threads:
        workflow.cores*0.45
    wildcard_constraints:
        parcellation = "LocaNMF"
    conda:
        "../envs/locaNMF_environment.yaml"

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
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "../scripts/trial_selection.py"

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
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,2000,1000)
    script:
        "../scripts/conditional.py"



rule feature_calculation:
    input:
        data = f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/data.h5",
        config = f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/conf.yaml",
    output:
        f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/features.h5",
        export_raw = report(
            f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.{config['export_type']}",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "{mouse_dates}", "Type": "Data"}),
        export_plot = report(
            f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.pdf",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "{mouse_dates}", "Type": "Plot"}),

        config = f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/conf.yaml",
    wildcard_constraints:
        feature = r'(?!thresh).+'
    params:
        params = lambda wildcards: config["features"][wildcards["feature"]]
    log:
        f"results/data/{{mouse_dates}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/feature_calculation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,2000)
    script:
        "../scripts/feature.py"

def concat_input(wildcards):

    sessions = subjects_from_wildcard(wildcards["concated_sessions"])

    return {"individual_sessions":[f"results/data/{'.'.join([subject_id,date])}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/features.h5" for subject_id,dates in sessions.items() for date in dates ]}

    
#if not config["combine_sessions"]:
# Sessions were not combined upon loading, concat individual sessions 
rule feature_concat:
    input:
        unpack(concat_input)
    output:
        f"results/data/{{concated_sessions}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/features.h5",
        export_raw = report(
            f"results/data/{{concated_sessions}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.{config['export_type']}",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "All", "Type": "Data"}),
        export_plot = report(
            f"results/data/{{concated_sessions}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.pdf",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "All", "Type": "Plot"}),  
    params:
        params = lambda wildcards: config["features"][wildcards["feature"]]
    wildcard_constraints:
        #only allowed to resolve wildcards of combined sessions (indicated by the sessions being concat with #) if set false in config, else sessions should be loaded together instead of being concat afterwards
        concated_sessions = r"GN[\w_.\-#]*" if not config["combine_sessions"] else r"(?!)" 
    log:
        f"results/data/{{concated_sessions}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/feature_calculation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,2000)
    script:
        "../scripts/feature_concat.py"
#else:
# Sessions were combined upon loading, no concatination


#Need prio over features
rule thresholding:
    input:
        data = f"{{data_dir}}/{{cond}}/{{feature}}/features.h5",
        config = f"{{data_dir}}/{{cond}}/{{feature}}/conf.yaml",
    output:
        data = f"{{data_dir}}/{{cond}}/{{feature}}_thresh~{{thresh}}/features.h5",
        export_raw = report(
            f"{{data_dir}}/{{cond}}/{{feature}}_thresh~{{thresh}}/features_thresh.{config['export_type']}",
            caption="../report/alignment.rst",
            category="5 Thresholding",
            subcategory="{feature}",
            labels={"Threshold": "{thresh}", "Condition": "{cond}", "Type": "Data"}),

        export_plot = report(
            f"{{data_dir}}/{{cond}}/{{feature}}_thresh~{{thresh}}/features_thresh.pdf",
            caption="../report/alignment.rst",
            category="5 Thresholding",
            subcategory="{feature}",
            labels={"Threshold": "{thresh}", "Condition": "{cond}", "Type": "Plot"}),
    params:
        params = lambda wildcards: config["features"][f"{wildcards['feature']}_thresh~{wildcards['thresh']}"]
    log:
        f"{{data_dir}}/{{cond}}/{{feature}}_thresh~{{thresh}}/feature_thresh.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,2000)
    script:
        "../scripts/thresholding.py"


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
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "../scripts/feature_elimination.py"

'''
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
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "../scripts/decoding.py"

'''

rule decoding:
    input:
        [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in config['trial_conditions']],
    output:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoder_model.pkl",
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoder_perf.pkl",
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoder_perf_across_timepoints.pkl",
        conf_m = f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/confusion_matrix.pkl",
        norm_conf_m = f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/norm_confusion_matrix.pkl",
        labels = f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/class_labels.pkl",
        config = f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    params:
        conds = list(config['trial_conditions']),
        params = lambda wildcards: config["decoders"][wildcards["decoder"]]
    log:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoding.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    threads:
        min(len(list(config['trial_conditions'])),workflow.cores) #For multinomial lr = job for each class
        #workflow.cores * 0.2
    script:
        "../scripts/decoding.py"

