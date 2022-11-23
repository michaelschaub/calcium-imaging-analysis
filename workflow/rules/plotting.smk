from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config


###   Plotting   ###

rule plot_parcels:
    '''
    decomposes data into different parcellations
    '''
    input:
        f"{{data_dir}}/{{parcellation}}/data.h5",
        config = f"{{data_dir}}/{{parcellation}}/conf.yaml",
    output:
        combined = report(
                    f"{{data_dir}}/{{parcellation}}/visualization/combined_parcels.png",
                    caption="../report/alignment.rst",
                    category="2 Parcellation",
                    subcategory="Overview",
                    labels={"Method":"{parcellation}"}),
        single = report(
                    directory(f"{{data_dir}}/{{parcellation}}/visualization/single_parcel/"),
                    patterns =["parcel_{name}.png"],
                    caption="../report/alignment.rst",
                    category="2 Parcellation",
                    subcategory="{parcellation}",
                    labels={"Parcel": "{name}",
                            "Method":"{parcellation}" }),
    params:
        n = config["parcels_n"],
    log:
        f"{{data_dir}}/{{parcellation}}/visualization/parcellation.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_parcels.py"

rule plot_performance:
    input:
        perf   = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in config["decoders"]],
        config = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{decoder}/conf.yaml" for decoder in config["decoders"]],
    output:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/performance.png",
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/performance.pkl",
    params:
        conds=list(config['trial_conditions']),
        decoders=config["decoders"],
    log:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/plot_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance.py"

rule plot_performances_features:
    input:
        perf   = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
                  for feature in config['features']
                  for decoder in config["decoders"]],
        config = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{feature}/{decoder}/conf.yaml"
                  for feature in config['features']
                  for decoder in config["decoders"]],
    output:
        report(f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/performances.png",
            caption="../report/decode_features.rst",
            category="6 Decoding",
            subcategory="Compare Features",
            labels={"Parcellation":"{parcellation}","Subject/Date": "{subject_dates}"}),
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/performances_anno.png"
    params:
        conds=list(config['trial_conditions']),
        decoders=config["decoders"],
        features=config['features'], #plot_feature_labels,
        subjects=config["plot_subject_labels"],
        trials=config['default_conditions'],
    log:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performances.py"

rule plot_performances_parcellations:
    input:
        [f"results/{{mouse_dates}}/{parcellation}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl"
         for parcellation in config['parcellations']
         for decoder in config["decoders"]],
    output:
        report(f"results/plots/{{mouse_dates}}/{{trials}}/Decoding/{'.'.join(config['trial_conditions'])}/{{feature}}/performances.png", caption="../report/decode_features.rst", category="6 Decoding", subcategory="Compare Parcellation", labels={"Feature":"{feature}"}),

        f"results/plots/{{mouse_dates}}/{{trials}}/Decoding/{'.'.join(config['trial_conditions'])}/{{feature}}/performances_anno.png",
    params:
        conds=list(config['trial_conditions']),
        decoders=config["decoders"],
        features=config['parcellations'], #plot_feature_labels,
        subjects=config["plot_subject_labels"],
        trials=config['default_conditions'],
    log:
        f"results/plots/{{mouse_dates}}/{{trials}}/Decoding/{'.'.join(config['trial_conditions'])}/{{feature}}/plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performances.py"

rule plot_performance_over_time:
    input:
        perf   = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in config["decoders"]],
        config = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{decoder}/conf.yaml" for decoder in config["decoders"]],
    output:
        #f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/{{subject_dates}}_{{parcellation}}_{{feature}}_{'.'.join(config['trial_conditions'])}_over_time.png",
        #f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/{{subject_dates}}_{{parcellation}}_{{feature}}_{'.'.join(config['trial_conditions'])}_over_time.pkl",
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/performance_over_time.png",
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/performance_over_time.pkl",
    params:
        conds=list(config['trial_conditions']),
        decoders=config["decoders"],
    log:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/plot_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_performance_matrix:
    input:
        perf   = f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/decoder_perf_across_timepoints.pkl",
        config = f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    output:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/{{decoder}}_performance_matrix.png",
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/{{decoder}}/performance_matrix_model.pkl",
    params:
        conds=list(config['trial_conditions']),
        decoders=config["decoders"],
    log:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/decoder/{'.'.join(config['trial_conditions'])}/{{feature}}/plots/{{decoder}}_plot_matrix_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_matrix.py"

rule plot_glassbrain:
    input:
        parcellation      = f"results/{{subject_dates}}/{{parcellation}}/data.h5",
        original_features = [f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Features/{cond}/{{feature}}/features.h5" for cond in config['trial_conditions']],
        features          = f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/best_feats.{config['export_type']}",

    output:
        plot              = f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/circle_plot.png",
        interactive_plot  = f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/glassbrain.html",
    log:
        f"results/{{subject_dates}}/{{parcellation}}/{{trials}}/Decoding/rfe/{'.'.join(config['trial_conditions'])}/{{rfe_n}}/{{feature}}/plot_glassbrain.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,16000,16000)
    script:
        "../scripts/plot_glassbrain.py"