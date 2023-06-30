from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config
from wildcard_functions import subjects_from_wildcard

### Todo unify simon and gerions inout functions (easy)


def sessions_input_gerion(wildcards):
    '''
    Matches the corresponding input files for one subject and date in the the {session_id} wildcard, task_flatten is only passed to make sure files are present. Rules uses the structured task files from the params.
    '''
    session_id = wildcards['session_id'].split('-')
    subject_id = session_id[0]
    date = '-'.join(session_id[1:])
    input = {
        "tasks": f"resources/experiment/{subject_id}/{date}/task_data/",
        "Vc": f"resources/experiment/{subject_id}/{date}/SVD_data/Vc_100s_highpass.mat", #TODO only works for new vc files, currently done this way to make sure only news vcs are used
        "trans_params": f"resources/experiment/{subject_id}/{date}/SVD_data/opts2.mat"}
    return input

def sessions_input_simon(wildcards):
    '''
    Matches the corresponding input files for one subject and date in the the {subject_date} wildcard, task_flatten is only passed to make sure files are present. Rules uses the structured task files from the params.
    '''
    session_id = wildcards['session_id'].split('-')
    subject_id = session_id[0]
    date = session_id[1:]
    input = {
        "tasks": f"resources/experiment/{subject_id}/{date}/SpatialDisc_Session.mat",
        "Vc": f"resources/experiment/{subject_id}/{date}/Vc.mat",
        "trans_params": f"resources/experiment/{subject_id}/{date}/opts2.mat"}
    return input

####

rule load_Random:
    output:
        "results/random.random/SVD/data.h5",
        config = "results/random.random/SVD/conf.yaml"
    log:
        f"results/random.random/SVD/random.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/random_data.py"     

rule load_GN:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        unpack(sessions_input_gerion)
    output:
        temp("results/data/{session_id}/SVD/{session_id}/data.h5"),
        align_plot = report("results/data/{session_id}/SVD/{session_id}/alignment.pdf", caption="../report/alignment.rst", category="1 Brain Alignment", labels={"Dataset": "GN", "Subjects":"{session_id}"}),
        config = "results/data/{session_id}/SVD/{session_id}/conf.yaml",
        stim_side = report("results/data/{session_id}/SVD/{session_id}/stim_side.pdf", caption="../report/alignment.rst", category="0 Loading", labels={"Dataset": "GN", "Subjects":"{session_id}"})
    wildcard_constraints:
        session_id = r"GN[\w_.\-]*"
    log:
        "results/data/{session_id}/SVD/{session_id}/pipeline_entry.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,1000)
    script:
        "../scripts/loading/load_GN.py"

#TODO fix this!
rule load_mSM:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        unpack(sessions_input_simon)
    output:
        temp("results/data/{session_id}/{session_id}/SVD/data.h5"),
        align_plot = report("results/data/{session_id}/SVD/{session_id}/alignment.pdf", caption="../report/alignment.rst", category="1 Brain Alignment", labels={"Dataset": "mSM", "Subjects":"{session_id}"}),
        config = "results/data/{session_id}/SVD/{session_id}/conf.yaml",
    wildcard_constraints:
        session_id = r"mSM[\w_.\-]*"
    log:
        "results/data/{session_id}/SVD/{session_id}/pipeline_entry.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,4000,1000)
    script:
        "../scripts/loading/load_mSM.py"

def input_unification(wildcards):
    dataset_id = wildcards['dataset_id']
    # if dataset_id is a defined alias, replace it with the canonical (hash) id, else leave it be
    if dataset_id in config['dataset_aliases']:
        dataset_id = config['dataset_aliases'].get(dataset_id)
        input = [ f"results/data/{dataset_id}/SVD/{dataset_id}/data.h5" ]
    else:
        digest_lng = config.get('hash_digest_length', 8)
        matched_ids = [ id for id in config['datasets'].keys() if dataset_id[:digest_lng] == id[:digest_lng] ]
        assert (len(matched_ids) == 1), f"Did not match exactly one dataset for {dataset_id=}, but instead {matched_ids=}"
        input = [ f"results/data/{subj}-{date}/SVD/{subj}-{date}/data.h5" for subj, date in config['datasets'].get(matched_ids[0])]
    return input

rule unify:
    '''
    Unifies multiple sessions into a single dataset, within a shared SVD space
    '''
    input:
        unpack(input_unification)
    output:
        "results/data/{dataset_id}/SVD/{dataset_id}/data.h5",
        config = "results/data/{dataset_id}/SVD/{dataset_id}/conf.yaml",
    log:
        "results/data/{dataset_id}/SVD/{dataset_id}/unify.log"
    params:
       alias=lambda wildcards: wildcards['dataset_id'] in config['dataset_aliases']
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,12000,4000)
    script:
        "../scripts/loading/unify_sessions.py"
