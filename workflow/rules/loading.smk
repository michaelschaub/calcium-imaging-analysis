from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config

def folder_from_wildcard(wildcard):
    subject_date_list = wildcard.split(".")
    subjects = subject_date_list[0::3]
    dates = subject_date_list[1::3]

    subject_dates = {}
    for i,s in enumerate(subjects):
        subject_dates[s] =  subject_dates.get(s,[])
        subject_dates[s].append(dates[i])

    return subject_dates

def sessions_input(wildcards):
    subject_dates_dict = folder_from_wildcard(wildcards["subject_dates"])
    input = {
        #To make sure that files are present, unfortunatly gets flattened -> losing information which dates belong to which subject
        "tasks_flatten":  [ f"resources/experiment/{subject_id}/{date}/task_data/" for subject_id,dates in subject_dates_dict.items() for date in dates ] ,
        "Vc": [  f"resources/experiment/{subject_id}/{date}/SVD_data/Vc.mat" for subject_id,dates in subject_dates_dict.items() for date in dates ],
        "trans_params": [ f"resources/experiment/{subject_id}/{date}/SVD_data/opts2.mat" for subject_id,dates in subject_dates_dict.items() for date in dates ]}
    return input

def sessions_params(wildcards):
    params = {
        "task_structured" : {subject_id: [ f"resources/experiment/{subject_id}/{date}/task_data/" for date in dates] for subject_id,dates in folder_from_wildcard(wildcards["subject_dates"]).items()},
    }
    return params

rule load_GN:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        unpack(sessions_input)
    output:
        "results/{subject_dates}/SVD/data.h5",
        align_plot = report("results/{subject_dates}/SVD/alignment.png", caption="../report/alignment.rst", category="1 Brain Alignment", labels={"Dataset": "GN", "Subjects":"{subject_dates}"}),
        config = "results/{subject_dates}/SVD/conf.yaml",
    params:
        sessions_params
    # so we are using this one and we can actually use a dict to make it even comfier
    #wildcard_constraints:
    #    subject_dates	= r"GN[a-zA-Z\d_.-]+",
    log:
        f"results/{{subject_dates}}/SVD/pipeline_entry.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,1000)
    script:
        "../scripts/default_entry.py"

rule load_mSM:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        sessions	= [ f"resources/experiment/{subject_id}/{date}/SpatialDisc_Session.mat"
                        for subject_id,dates in config["subjects"].items() for date in dates],
        Vc		= [ f"resources/experiment/{subject_id}/{date}/Vc.mat"
                      for subject_id,dates in config["subjects"].items() for date in dates],
        trans_params	= [ f"resources/experiment/{subject_id}/{date}/opts2.mat"
                            for subject_id,dates in config["subjects"].items() for date in dates],
    output:
        f"results/{{subject_dates}}/SVD/data.h5",
        align_plot = report("results/{subject_dates}/SVD/alignment.png", caption="../report/alignment.rst", category="1 Brain Alignment", labels={"Dataset": "mSM", "Subjects":", ".join(config["subjects"])}),
        config = f"results/{{subject_dates}}/SVD/conf.yaml",
    params:
        subject_dates_str = '_'.join(config["subject_dates"]),
        sessions_structured = {subject_id: { date: f"resources/experiment/{subject_id}/{date}/SpatialDisc_Session.mat"
                                             for date in dates} for subject_id,dates in config["subjects"].items()}
    wildcard_constraints:
        subject_dates	= r"mSM[a-zA-Z\d_-]+",
    log:
        f"results/{{subject_dates}}/SVD/pipeline_entry.log"
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,1000)
    script:
        "../scripts/mSM_entry.py"