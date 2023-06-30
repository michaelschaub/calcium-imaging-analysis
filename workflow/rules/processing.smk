from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config, temp_if_config
from wildcard_functions import subjects_from_wildcard

import re

def temp_c(file, rule=None):
    return temp_if_config(file, config.get("temporary_outputs",{}), rule)

###   Data processing   ###

def parcellation_input(wildcards):

    if not bool(re.match(r"(?!SVD$).+",wildcards["parcellation"])):  #TODO why is snakemake regex broken? :( ,fro now evaulate the same freaking expression within input function and require non existing files to exclude this rule....
        raise ValueError("This is not the correct rule for this!")
        return {"data":f"good/luck/finding/this/non/existing/path"}

    input = {
        "data"	: "{data_dir}/{dataset_id}/SVD/{dataset_id}/data.h5",
        "config": "{data_dir}/{dataset_id}/SVD/{dataset_id}/conf.yaml" }
    branch = config["parcellations"][wildcards["parcellation"]]["branch"]
    input.update( config["parcellation_wildcard_matching"][branch] )
    if wildcards['parcellation'] == "SVD":
        raise ValueError("This is not the correct rule for this!")
    return input

rule parcellate:
    '''
    decomposes data into different parcellations
    '''
    input:
        unpack(parcellation_input)
    output:
        temp_c("{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/data.h5", rule="parcellate"),
        config = "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/conf.yaml",
    params:
        params = lambda wildcards: config["parcellations"][wildcards["parcellation"]]
    wildcard_constraints:
        # exclude SVD as parcellation
        #TODO check if this really works
        parcellation = "(?!SVD$).+",
	#data_dir = 'results/data/GN06.03-26#GN06.03-29'
    log:
        "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/parcellation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,2000)
    script:
        "../scripts/parcellation.py"

use rule parcellate as locaNMF with:
    threads:
        workflow.cores*0.45
    wildcard_constraints:
        parcellation = "LocaNMF"
    conda:
        "../envs/locaNMF_environment.yaml"

def input_trial_selection(wildcards):
    dataset_id = wildcards['dataset_id']
    selection_id = wildcards["selection_id"]
    if dataset_id == selection_id or '/' in selection_id:
        raise ValueError("Trial selection can not produce these outputs.")
    if selection_id in config['dataset_aliases']:
        selection_id = config['dataset_aliases'].get(selection_id)
        input = [ f"results/data/{{dataset_id}}/{{parcellation}}/{selection_id}/data.h5" ]
    else:
        input = [ "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/data.h5" ]
    return input

def trial_selection_params(wildcards):
    selection_id = wildcards["selection_id"]
    if selection_id in config['dataset_aliases']:
        selection_id = config['dataset_aliases'].get(selection_id)
        alias = True
    else:
        alias = False
    params = config["trial_selection"].get(selection_id, {'branch':selection_id, 'is_dataset':False})
    params['alias'] = alias
    return params

rule trial_selection:
    '''
    can select trials from a dataset and (TODO) apply through predefined filters
    '''
    input:
        #data = "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/data.h5",
        #config = "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/conf.yaml",
        input_trial_selection
    output:
        temp_c("{data_dir}/{dataset_id}/{parcellation}/{selection_id}/data.h5", rule="trial_selection"),
        #report = report("{data_dir}/{dataset_id}/{parcellation}/{selection_id}/conf.yaml"),
        config = "{data_dir}/{dataset_id}/{parcellation}/{selection_id}/conf.yaml",
    params:
        trial_selection_params
    log:
        "{data_dir}/{dataset_id}/{parcellation}/{selection_id}/trial_selection.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,2000,2000)
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
        temp_c(f"{{data_dir}}/Features/{{cond}}/data.h5", rule="condition"),
        config = f"{{data_dir}}/Features/{{cond}}/conf.yaml",
    params:
        condition_params
    wildcard_constraints:
        cond = branch_match(list(config['trial_conditions'].keys()), params=False)
    log:
        f"{{data_dir}}/Features/{{cond}}/conditionals.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,2000,2000)
    script:
        "../scripts/conditional.py"

rule condition_grouping:
    input:
        lambda wildcards: [
		f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{sub_cond}/data.h5"
		for sub_cond in config['group_conditions'][wildcards['cond']]
	]
    output:
        temp_c(f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/data.h5", rule="condition"),
        config = f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/conf.yaml",
    wildcard_constraints:
        cond = branch_match(list(config['group_conditions'].keys()), params=False)
    params:
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/condition_grouping.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,2000)
    script:
        "../scripts/group_conditions.py"

rule feature_calculation:
    input:
        data = f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/data.h5",
        config = f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/conf.yaml",
    output:
        temp_c(f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/features.h5", rule="feature"),
        export_raw = report(
            f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.{config['export_type']}",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "{dataset_id}", "Type": "Data"}),
        export_plot = report(
            f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.pdf",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "{dataset_id}", "Type": "Plot"}),

        config = f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/conf.yaml",
    wildcard_constraints:
        feature = r'(?!thresh).+'
    params:
        params = lambda wildcards: config["features"][wildcards["feature"]]
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/feature_calculation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,4000)
    script:
        "../scripts/feature.py"

def concat_input(wildcards):

    sessions = subjects_from_wildcard(wildcards["concated_sessions"])

    return {"individual_sessions":[f"results/data/{'.'.join([subject_id,date])}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/features.h5" for subject_id,dates in sessions.items() for date in dates ]}

    
# can this actually occur with new structure? 
# TODO remove?
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
        concated_sessions = r"GN[\w_.\-#]*" if not True else r"(?!)"
    log:
        f"results/data/{{concated_sessions}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/feature_calculation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,2000)
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
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,2000)
    script:
        "../scripts/thresholding.py"

rule feature_grouping:
    input:
        lambda wildcards: [
		f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{sub_cond}/{{feature}}/features.h5"
		for sub_cond in config['group_conditions'][wildcards['cond']]
	]
    output:
        temp_c(f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/features.h5", rule="feature"),
        export_raw = report(
            f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.{config['export_type']}",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "{dataset_id}", "Type": "Data"}),
        export_plot = report(
            f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/{{cond}}.{{feature}}.pdf",
            caption="../report/alignment.rst",
            category="4 Feature Calculation",
            subcategory="{feature}",
            labels={"Condition": "{cond}", "Subject/Date": "{dataset_id}", "Type": "Plot"}),

        config = f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/conf.yaml",
    wildcard_constraints:
        cond = branch_match(list(config['group_conditions'].keys()), params=False)
    params:
        params = lambda wildcards: config["features"][wildcards["feature"]]
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{trials}}/Features/{{cond}}/{{feature}}/feature_grouping.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,2000)
    script:
        "../scripts/group_features.py"


rule feature_elimination:
    input:
        feats = [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
    output:
        best_feats	= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/best_feats.{config['export_type']}",
        model		= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/decoder_model.pkl",
        perf		= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/decoder_perf.{config['export_type']}",
        config		= f"{{data_dir}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/conf.yaml",
    params:
        conds = list(config["aggr_conditions"]),
        reps = config['feature_selection']['reps']
    log:
        f"{{data_dir}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/feature_calculation.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,4000)
    script:
        "../scripts/feature_elimination.py"

'''
rule decoding:
    input:
        [f"{{data_dir}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
    output:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_model.pkl",
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_perf.pkl",
        config = f"{{data_dir}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    params:
        conds = list(config['aggr_conditions']),
        params = lambda wildcards: config["decoders"][wildcards["decoder"]]
    log:
        f"{{data_dir}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoding.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,1000,1000)
    script:
        "../scripts/decoding.py"

'''

rule decoding:
    input:
        [f"{{data_dir}}/{{parcellation}}/{{trials}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
    output:
        f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_model.pkl",
        f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_perf.pkl",
        f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_perf_across_timepoints.pkl",
        conf_m = f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/confusion_matrix.pkl",
        norm_conf_m = f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/norm_confusion_matrix.pkl",
        labels = f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/class_labels.pkl",
        df = f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/perf_df.pkl",
        config = f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    params:
        conds = list(config['aggr_conditions']),
        params = lambda wildcards: config["decoders"][wildcards["decoder"]]
    log:
        f"{{data_dir}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoding.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,8000,4000)
    threads:
        min(len(list(config['aggr_conditions'])),workflow.cores) #For multinomial lr = job for each class
        #workflow.cores * 0.2
    script:
        "../scripts/decoding.py"

