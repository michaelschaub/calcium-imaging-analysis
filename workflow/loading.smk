from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config

rule load_GN:
    '''
    aggregates all task and svd data from one session with one animal
    '''
    input:
        #To make sure that files are present, unfortunatly gets flattened -> losing information which dates belong to which subject
        _		= [[ f"resources/experiment/{subject_id}/{date}/task_data/"
                      for date in dates] for subject_id,dates in config["subjects"].items()],
        Vc		= [ f"resources/experiment/{subject_id}/{date}/SVD_data/Vc.mat"
                      for subject_id,dates in config["subjects"].items() for date in dates],
        trans_params	= [ f"resources/experiment/{subject_id}/{date}/SVD_data/opts.mat"
                            for subject_id,dates in config["subjects"].items() for date in dates],
    output:
        f"results/{{subject_dates}}/SVD/data.h5",
        align_plot = report("results/{subject_dates}/SVD/alignment.png", caption="report/alignment.rst", category="1 Brain Alignment", labels={"Dataset": "GN", "Subjects":", ".join(config["subjects"])}),
        config = f"results/{{subject_dates}}/SVD/conf.yaml",
    params:
        subject_dates_str = '_'.join(config["subject_dates"]),
        #maybe find a clean solution from flattened array,
        task_structured = {subject_id: [ f"resources/experiment/{subject_id}/{date}/task_data/"
                                         for date in dates] for subject_id,dates in config["subjects"].items()} # so we are using this one and we can actually use a dict to make it even comfier
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
        align_plot = report("results/{subject_dates}/SVD/alignment.png", caption="report/alignment.rst", category="1 Brain Alignment", labels={"Dataset": "mSM", "Subjects":", ".join(config["subjects"])}),
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
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,4000,1000)
    script:
        "scripts/mSM_entry.py"
