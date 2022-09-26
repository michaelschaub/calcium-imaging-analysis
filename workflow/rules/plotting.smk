###   Plotting   ###

rule plot_parcels:
    '''
    decomposes data into different parcellations
    '''
    input:
        f"{{data_dir}}/{{parcellation}}/data.h5",
        config = f"{{data_dir}}/{{parcellation}}/conf.yaml",
    output:
        combined = f"{{data_dir}}/{{parcellation}}/visualization/combined_parcels.png",
        single = directory(f"{{data_dir}}/{{parcellation}}/visualization/single_parcel/")
    params:
        n = config['branch_opts']['plotting']['plot_parcels']['n']
    log:
        f"{{data_dir}}/{{parcellation}}/visualization/parcellation.log"
    conda:
        "envs/environment.yaml"
    script:
        "scripts/plot_parcels.py"

rule plot_performance:
    input:
        perf   = [f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in decoders],
        config = [f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{decoder}/conf.yaml" for decoder in decoders],
    output:
        f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/plots/performance.png",
        f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/plots/performance.pkl",
    params:
        conds=list(trial_conditions),
        decoders=decoders,
    log:
        f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/plots/plot_performance.log",
    conda:
        "envs/environment.yaml"
    script:
        "scripts/plot_performance.py"

rule plot_performances:
    input:
        perf   = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{feature}/{decoder}/decoder_perf.pkl"
                  for feature in features
                  for decoder in decoders],
        config = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{feature}/{decoder}/conf.yaml"
                  for feature in features
                  for decoder in decoders],
    output:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/performances.png",
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/performances_anno.png",
    params:
        conds=list(trial_conditions),
        decoders=decoders,
        features=features, #plot_feature_labels,
        subjects=plot_subject_labels,
        trials=default_conditions,
    log:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/plot_performances.log",
    conda:
        "envs/environment.yaml"
    script:
        "scripts/plot_performances.py"

rule plot_performances_parcellations:
    input:
        [f"results/{{mouse_dates}}/{parcellation}/{{trials}}/Decoding/decoder/{'.'.join(trial_conditions)}/{{feature}}/{decoder}/decoder_perf.pkl"
         for parcellation in parcellations
         for decoder in decoders],
    output:
        f"results/plots/{{mouse_dates}}/{{trials}}/Decoding/{'.'.join(trial_conditions)}/{{feature}}/performances.png",
        f"results/plots/{{mouse_dates}}/{{trials}}/Decoding/{'.'.join(trial_conditions)}/{{feature}}/performances_anno.png",
    params:
        conds=list(trial_conditions),
        decoders=decoders,
        features=parcellations, #plot_feature_labels,
        subjects=plot_subject_labels,
        trials=default_conditions,
    log:
        f"results/plots/{{mouse_dates}}/{{trials}}/Decoding/{'.'.join(trial_conditions)}/{{feature}}/plot_performances.log",
    conda:
        "envs/environment.yaml"
    script:
        "scripts/plot_performances.py"


rule plot_glassbrain:
    input:
        parcellation      = f"results/{subject_dates}/{{parcellation}}/data.h5",
        original_features = [f"results/{subject_dates}/{{parcellation}}/{{trials}}/Features/{cond}/{{feature}}/features.h5" for cond in trial_conditions],
        features          = f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/best_feats.{export_type}",

    output:
        plot              = f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/circle_plot.png",
        interactive_plot  = f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/glassbrain.html",
    log:
        f"results/{subject_dates}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(trial_conditions)}/{{rfe_n}}/{{feature}}/plot_glassbrain.log",
    conda:
        "envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,16000,16000)
    script:
        "scripts/plot_glassbrain.py"