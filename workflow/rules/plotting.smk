from snakemake_tools import calculate_memory_resource as mem_res

include: "common.smk"

###   Plotting   ###

rule plot_from_df_subset_space_comp:
    input:
        f"{PLOTS_DIR}/{{dataset_id}}/sessions_in_sessions/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
        f"{PLOTS_DIR}/{{dataset_id}}/sessions_in_subsets/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
        f"{PLOTS_DIR}/{{dataset_id}}/sessions_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
        f"{PLOTS_DIR}/{{dataset_id}}/subsets_in_subsets/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
        f"{PLOTS_DIR}/{{dataset_id}}/subsets_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
        f"{PLOTS_DIR}/{{dataset_id}}/dataset_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/perf_subset_space_comp.pdf",
    params:
        #style = 'dataset_id',
        hue  = 'aggregated_from',
        #size
        conds=list(config['aggr_conditions']), #maybe get from config within script?
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/perf_subset_space_comp.log",
    conda:
        "../envs/environment.yaml"
    priority:
        50
    script:
        "../scripts/plotting/plot_performance_time_aggr.py"


rule plot_from_df_all_space_comp:
    input:
        [ f"{PLOTS_DIR}/{dataset_id}/sessions_in_sessions/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
            for dataset_id in config["session_runs"].keys() if dataset_id != config['dataset_aliases']['All']],
        [ f"{PLOTS_DIR}/{dataset_id}/sessions_in_subsets/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
            for dataset_id in config["session_runs"].keys() if dataset_id != config['dataset_aliases']['All']],
        [ f"{PLOTS_DIR}/{dataset_id}/subsets_in_subsets/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
            for dataset_id in config["session_runs"].keys() if dataset_id != config['dataset_aliases']['All']],
        [ f"{PLOTS_DIR}/{dataset_id}/sessions_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
            for dataset_id in config["session_runs"].keys() if dataset_id != config['dataset_aliases']['All']],
        [ f"{PLOTS_DIR}/{dataset_id}/subsets_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
            for dataset_id in config["session_runs"].keys() if dataset_id != config['dataset_aliases']['All']],
        [ f"{PLOTS_DIR}/{dataset_id}/dataset_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
            for dataset_id in config["session_runs"].keys() if dataset_id != config['dataset_aliases']['All']],
        f"{PLOTS_DIR}/{config['dataset_aliases']['All']}/subsets_in_dataset/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl",
    output:
        f"{PLOTS_DIR}/{config['dataset_aliases']['All']}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/perf_all_space_comp.pdf",
    params:
        #style = 'dataset_id',
        hue  = 'aggregated_from',
        #size
        conds=list(config['aggr_conditions']), #maybe get from config within script?
    log:
        f"{PLOTS_DIR}/{config['dataset_aliases']['All']}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/perf_all.datasets.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/plot_performance_time_aggr.py"

def plot_generalization_input(wildcards):
    perf_path = "{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{testing_set_id}/Decoding/decoder/{{conditions}}/{{feature}}/{{decoder}}/model_from/{decoding_set_id}/{session_id}/cluster_perf_df.pkl"
    cross_dataset_perfs = [ perf_path.format(DATA_DIR=DATA_DIR,
                                             testing_set_id=generalization['testing_set_id'],
                                             decoding_set_id=generalization['decoding_set_id'],
                                             session_id=session_id)
                                for generalization in config['generalize'][wildcards['dataset_id']]
                                for session_id in config['dataset_sessions'][generalization['testing_set_id']]]
    inner_dataset_perfs = [ perf_path.format(DATA_DIR=DATA_DIR,
                                             testing_set_id=generalization['decoding_set_id'],
                                             decoding_set_id=generalization['decoding_set_id'],
                                             session_id=session_id)
                                for generalization in config['generalize'][wildcards['dataset_id']]
                                for session_id in config['dataset_sessions'][generalization['decoding_set_id']]]
    return [*cross_dataset_perfs , *inner_dataset_perfs]

rule plot_generalization:
    input:
        plot_generalization_input
    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/generalized.pdf"
    params:
        #hue   = 'model',
        hue  = ("{decoding_set_id} on {testing_set_id}", ['decoding_set_id', 'testing_set_id']),
        x     = 'date_time',
        conds=list(config['aggr_conditions']), #maybe get from config within script?
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/generalized.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/plot_performance_time_aggr.py"



# noinspection SmkNotSameWildcardsSet
rule plot_parcels:
    '''
    plots spatial compoenents of parcellation 
    '''
    input:
        "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/data.h5",
        config = "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/conf.yaml",
    output:
        combined = report(
                    "{data_dir}/{dataset_id}/{parcellation}/visualization/combined_parcels.pdf",
                    caption="../report/alignment.rst",
                    category="2 Parcellation",
                    subcategory="Overview",
                    labels={"Method":"{parcellation}"}),
        single = report(
                    directory("{data_dir}/{dataset_id}/{parcellation}/visualization/single_parcel/"),
                    patterns =["parcel_{name}.pdf"],
                    caption="../report/alignment.rst",
                    category="2 Parcellation",
                    subcategory="{parcellation}",
                    labels={"Parcel": "{name}",
                            "Method":"{parcellation}" }),
    params:
        n = config["parcels_n"],
    log:
        "{data_dir}/{dataset_id}/{parcellation}/visualization/parcellation.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_parcels.py"

rule plot_model_coef:
    input:
        model  = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{{conditions}}/{{feature}}/{{decoder}}/decoder_model.pkl",
        parcel = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{dataset_id}}/data.h5"
    output:
        coef_plot = f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/model_coef_mean.pdf",
        var_plot = f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/model_coef_std.pdf"
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/decoder_model.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_model_coef.py"

######

rule plot_performances_clusters_time:
    input:
        perf   =[f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/cluster_perf.pkl"],

    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.pdf"
        
        

        #report(f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_perf.pdf",
        #    caption="../report/decode_features.rst",
        #    category="6 Decoding",
        #    subcategory="Compare Features",
        #    labels={"Parcellation":"{parcellation}","Subject/Date": "{dataset_id}"}),
        #f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/performances_anno.pdf"
    params:
        clusters="",
        decoders=[f"{{decoder}}"],
        conds=list(config['aggr_conditions']),
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.log"
        
    conda:
        "../envs/environment.yaml"
    script:
       "../scripts/plot_performance_time.py"


rule plot_performances_clusters_time_different_subject:
    input:
        perf = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{testing_set_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/model_from/{{decoding_set_id}}/cluster_perf.pkl"]
    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{decoding_set_id}}_on_{{testing_set_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.pdf"

        #report(f"{PLOTS_DIR}/{{dataset_id}}/{{decoding_set_id}}_on_{{testing_set_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_perf.pdf",
        #    caption="../report/decode_features.rst",
        #    category="6 Decoding",
        #    subcategory="Compare Features",
        #    labels={"Parcellation":"{parcellation}","Subject/Date": "{dataset_id}"}),
        #f"{PLOTS_DIR}/{{dataset_id}}/{{decoding_set_id}}_on_{{testing_set_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_performances_anno.pdf"
    params:
        clusters="",
        decoders=[f"{{decoder}}"],
        conds=list(config['aggr_conditions']),
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{decoding_set_id}}_on_{{testing_set_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.log"
        
    conda:
        "../envs/environment.yaml"
    script:
       "../scripts/plot_performance_time.py"

    
########

rule plot_performance:
    input:
        perf   = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in config["decoders"]],
        config = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/conf.yaml" for decoder in config["decoders"]],
    output:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/performance.pdf",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
    log:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/plot_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance.py"

rule plot_performances_features:
    input:
        perf   = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
                  for feature in config['features']
                  for decoder in config["decoders"]],
        config = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/conf.yaml"
                  for feature in config['features']
                  for decoder in config["decoders"]],
    output:
        report(f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_perf.pdf",
            caption="../report/decode_features.rst",
            category="6 Decoding",
            subcategory="Compare Features",
            labels={"Parcellation":"{parcellation}","Subject/Date": "{dataset_id}"}),
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/performances_anno.pdf"
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        features=config['features'], #plot_feature_labels,
        subjects=config["plot_subject_labels"],
        trials=config['default_conditions'],
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performances.py"

rule plot_performances_parcellations:
    input:
        [f"{DATA_DIR}/{{dataset_id}}/{parcellation}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl"
         for parcellation in config['parcellations']
         for decoder in config["decoders"]],
    output:
        report(f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}_perf.pdf", caption="../report/decode_features.rst", category="6 Decoding", subcategory="Compare Parcellation", labels={"Feature":"{feature}"}),

        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/performances_anno.pdf",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        features=config['parcellations'], #plot_feature_labels,
        subjects=config["plot_subject_labels"],
        trials=config['default_conditions'],
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}_plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performances.py"


rule plot_condition_diff:
    input:
        [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Features/{cond}/data.h5" for cond in config['trial_conditions']],
    output:
        directory(f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['trial_conditions'])}/b-maps/{{parcellation}}/Phases/"), #Do not replace with aggr_condtiions
        static = f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['trial_conditions'])}/b-maps/{{parcellation}}/all_cond_difference_b-maps.pdf"
    params:
        conds = config['trial_conditions']
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['trial_conditions'])}/b-maps/{{parcellation}}/cond_difference_b-maps.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/plot_cond_b-maps.py"

rule plot_performances_features_parcellations:
    input:
        [f"{DATA_DIR}/{{dataset_id}}/{parcellation}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
            for feature in config['features']
            for parcellation in config['parcellations']
            for decoder in config["decoders"]]
    output:
        report(f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/all_perf.pdf",
            caption="../report/decode_features.rst",
            category="6 Decoding",
            subcategory="Compare Features & Parcellation",
            labels={"Subject/Date": "{dataset_id}"}),
        
    params:
        conds=list(config['aggr_conditions']),
        features=config['features'],
        parcellations=config['parcellations'],
        decoders=config["decoders"],
        #plot_feature_labels,
        #subjects=config["plot_subject_labels"],
        #trials=config['default_conditions'],
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_violin.py"

rule plot_performance_over_time:
    input:
        perf   = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in config["decoders"]],
        config = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/conf.yaml" for decoder in config["decoders"]],
    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/performance_over_time.pdf",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{{parcellation}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/plot_performance_over_time.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_performance_matrix:
    input:
        perf   = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_perf_across_timepoints.pkl",
        config = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    output:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_performance_matrix.pdf",
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/performance_matrix.pdf",
        cluster = f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/performance_clustered.pdf",
        model = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}/performance_matrix_model.pkl",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
    log:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_plot_matrix_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_matrix.py"

rule plot_glassbrain:
    input:
        parcellation      = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{dataset_id}}/data.h5",
        original_features = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
        features          = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/best_feats.{config['export_type']}",

    output:
        plot              = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/circle_plot.pdf",
        interactive_plot  = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/glassbrain.html",
    log:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/plot_glassbrain.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mib=lambda wildcards, input, attempt: mem_res(wildcards,input,attempt,16000,16000)
    script:
        "../scripts/plot_glassbrain.py"



rule plot_performances_parcellations_over_time:
    input:
        perf = [f"{DATA_DIR}/{{dataset_id}}/{parcellation}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl"
         for decoder in config["decoders"]
         for parcellation in config['parcellations']]
    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}_performance_over_time.pdf",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        parcellations=list(config['parcellations']), #plot_feature_labels,
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{feature}}_performances_over_time.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_performances_features_over_time:
    input:
        perf = [f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
         for decoder in config["decoders"]
         for feature in config['features']]
    output:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_performance_over_time.pdf",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        features=list(config['features']), #plot_feature_labels,
    log:
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}/{{parcellation}}_performances_over_time.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_conf_matrix:
    input:
        conf_matrix = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/confusion_matrix.pkl",
        norm_conf_matrix = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/norm_confusion_matrix.pkl",
        class_labels = f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/class_labels.pkl",
    output:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_confusion_matrix.pdf",
        f"{PLOTS_DIR}/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/confusion_matrix.pdf",
    params:
        phases = config["phase_conditions"]
    log:
        f"{DATA_DIR}/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_confusion_matrix.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_confusion_matrix.py"

rule plot_activity:
    input:
        "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/data.h5"
    output:
        "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/activity.pdf"
    log:
         "{data_dir}/{dataset_id}/{parcellation}/{dataset_id}/activity.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_activity.py"
