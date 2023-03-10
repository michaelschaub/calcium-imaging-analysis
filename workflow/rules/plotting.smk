from snakemake_tools import create_parameters, create_conditions, calculate_memory_resource as mem_res, branch_match, hash_config, readable_dataset_id


###   Plotting   ###

rule ind_aggregate_results_as_df:
    input:
        lambda wildcards: [f"results/data/{{dataset_id}}/{{parcellation}}/{subset_id}/Decoding/decoder/{{conditions}}/{{feature}}/{{decoder}}/perf_df.pkl"
            for subset_id in config["session_runs"][wildcards["dataset_id"]] ]
    output:
        f"results/plots/{{dataset_id}}/ind_decode/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl"
    log:
        f"results/plots/{{dataset_id}}/ind_decode/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/df_aggregation.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/aggregate_results.py"

use rule ind_aggregate_results_as_df as shared_aggregate_results_as_df with:
    input:
        "results/data/{dataset_id}/{parcellation}/{dataset_id}/Decoding/decoder/{conditions}/{feature}/{decoder}/perf_df.pkl"
    output:
        "results/plots/{dataset_id}/shared_decode/{conditions}/{feature}/{parcellation}/{decoder}/aggr_perf_df.pkl"
    log:
        f"results/plots/{{dataset_id}}/shared_decode/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/df_aggregation.log"

#rule plot_from_df_dataset:
#    input:
#        "results/plots/{dataset_hash}/ind_decode/{conditions}/{feature}/{parcellation}/{decoder}/aggr_perf_df.pkl"
#        "results/plots/{dataset_hash}/shared_decode/{conditions}/{feature}/{parcellation}/{decoder}/aggr_perf_df.pkl" 
#    output:
#        "results/plots/{dataset_hash}/{conditions}/{feature}/{parcellation}/{decoder}/perf_all.datasets.pdf"

rule plot_from_df_all:
    input:
        [f"results/plots/{dataset_hash}/ind_decode/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl" for dataset_hash in config["session_runs"] ],
        [f"results/plots/{dataset_hash}/shared_decode/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/aggr_perf_df.pkl" for dataset_hash in config["session_runs"] ],
    output:
        f"results/plots/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/perf_all.datasets.pdf",
    log:
        "results/plots/{conditions}/{feature}/{parcellation}/{decoder}/perf_all.datasets.log",
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
        model  = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{{conditions}}/{{feature}}/{{decoder}}/decoder_model.pkl",
        parcel = f"results/data/{{dataset_id}}/{{parcellation}}/{{dataset_id}}/data.h5"
    output:
        coef_plot = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/model_coef_mean.pdf",
        var_plot = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/model_coef_std.pdf"
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/decoder_model.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_model_coef.py"

### move to processing
rule cluster_coef:
    input:
        model  = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{{conditions}}/{{feature}}/{{decoder}}/decoder_model.pkl",
        parcel = f"results/data/{{dataset_id}}/{{parcellation}}/{{dataset_id}}/data.h5"
    output:
        no_cluster = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/CoefsAcrossTime.pdf",
        cluster    = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/CoefsAcrossTime_clustered.pdf",
        PCA_3D = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/CoefsAcrossTime_PCA_3D.html",
        Cluster_3D = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/CoefsAcrossTime_Clustered_3D.html",

        cluster_small    = f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/CoefsAcrossTime_clustered.png",
        models     = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{{conditions}}/{{feature}}/{{decoder}}/ClusterModels.pkl",
        coef_plots  = directory(f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/Clusters/")
    params:
        phases = config["phase_conditions"]
    log:
         f"results/plots/{{dataset_id}}/{{subset_id}}/{{conditions}}/{{feature}}/{{parcellation}}/{{decoder}}/CoefsAcrossTime_clustered.log",
    conda:
        "../envs/environment_clustering.yaml"
    script:
        "../scripts/cluster_models_UMAP.py"


rule decoding_with_existing_model:
    input:
        feat = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
        models = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/ClusterModels.pkl",
    output:
        perf =f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/cluster_perf.pkl"
    params:
        conds = list(config['aggr_conditions']),
        decoders=[f"{{decoder}}"],
        params = lambda wildcards: config["decoders"][wildcards["decoder"]], #TODO actually we just need number of reps, or we could also just test once on whole dataset
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/cluster_perf.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "../scripts/test_models.py"

#TODO do properly

rule decoding_with_existing_model_different_subject:
    input:
        feat = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
        models = f"results/data/{config['generalize_from']}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/ClusterModels.pkl",
        org_decomp = f"results/data/{config['generalize_from']}/{{parcellation}}/data.h5",
        new_decomp = f"results/data/{{dataset_id}}/{{parcellation}}/{{dataset_id}}/data.h5",
    output:
        perf =f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/from_{config['generalize_from']}/cluster_perf.pkl"
    params:
        conds = list(config['aggr_conditions']),

        decoders=[f"{{decoder}}"],
        params = lambda wildcards: config["decoders"][wildcards["decoder"]], #TODO actually we just need number of reps, or we could also just test once on whole dataset
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/from_{config['generalize_from']}/cluster_perf.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,1000,1000)
    script:
        "../scripts/test_models.py"
######

rule plot_performances_clusters_time:
    input:
        perf   =[f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/cluster_perf.pkl"],

    output:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.pdf"
        
        

        #report(f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_perf.pdf",
        #    caption="../report/decode_features.rst",
        #    category="6 Decoding",
        #    subcategory="Compare Features",
        #    labels={"Parcellation":"{parcellation}","Subject/Date": "{dataset_id}"}),
        #f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/performances_anno.pdf"
    params:
        clusters="",
        decoders=[f"{{decoder}}"],
        conds=list(config['aggr_conditions']),
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.log"
        
    conda:
        "../envs/environment.yaml"
    script:
       "../scripts/plot_performance_time.py"


rule plot_performances_clusters_time_different_subject:
    input:
        perf   =[f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/from_{config['generalize_from']}/cluster_perf.pkl"],
    output:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/from_{config['generalize_from']}/ClusturedModels_perf.pdf"
        

        #report(f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_perf.pdf",
        #    caption="../report/decode_features.rst",
        #    category="6 Decoding",
        #    subcategory="Compare Features",
        #    labels={"Parcellation":"{parcellation}","Subject/Date": "{dataset_id}"}),
        #f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/performances_anno.pdf"
    params:
        clusters="",
        decoders=[f"{{decoder}}"],
        conds=list(config['aggr_conditions']),
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/ClusturedModels_perf.log"
        
    conda:
        "../envs/environment.yaml"
    script:
       "../scripts/plot_performance_time.py"

    
########

rule plot_performance:
    input:
        perf   = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in config["decoders"]],
        config = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/conf.yaml" for decoder in config["decoders"]],
    output:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/performance.pdf",
        
        #f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/performance.pkl",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/plot_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance.py"

rule plot_performances_features:
    input:
        perf   = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
                  for feature in config['features']
                  for decoder in config["decoders"]],
        config = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/conf.yaml"
                  for feature in config['features']
                  for decoder in config["decoders"]],
    output:
        report(f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_perf.pdf",
            caption="../report/decode_features.rst",
            category="6 Decoding",
            subcategory="Compare Features",
            labels={"Parcellation":"{parcellation}","Subject/Date": "{dataset_id}"}),
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/performances_anno.pdf"
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        features=config['features'], #plot_feature_labels,
        subjects=config["plot_subject_labels"],
        trials=config['default_conditions'],
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performances.py"

rule plot_performances_parcellations:
    input:
        [f"results/data/{{dataset_id}}/{parcellation}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl"
         for parcellation in config['parcellations']
         for decoder in config["decoders"]],
    output:
        report(f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}_perf.pdf", caption="../report/decode_features.rst", category="6 Decoding", subcategory="Compare Parcellation", labels={"Feature":"{feature}"}),

        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/performances_anno.pdf",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        features=config['parcellations'], #plot_feature_labels,
        subjects=config["plot_subject_labels"],
        trials=config['default_conditions'],
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{{feature}}/{'.'.join(config['aggr_conditions'])}/plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performances.py"


rule plot_condition_diff:
    input:
        [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Features/{cond}/data.h5" for cond in config['trial_conditions']],
    output:
        directory(f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['trial_conditions'])}/b-maps/{{parcellation}}/Phases/"), #Do not replace with aggr_condtiions
        static = f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['trial_conditions'])}/b-maps/{{parcellation}}/all_cond_difference_b-maps.pdf"
    params:
        conds = config['trial_conditions']
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['trial_conditions'])}/b-maps/{{parcellation}}/cond_difference_b-maps.log"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plotting/plot_cond_b-maps.py"

rule plot_performances_features_parcellations:
    input:
        [f"results/data/{{dataset_id}}/{parcellation}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
            for feature in config['features']
            for parcellation in config['parcellations']
            for decoder in config["decoders"]]
    output:
        report(f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/all_perf.pdf",
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
        f"results/data/{{dataset_id}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/plot_performances.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_violin.py"

rule plot_performance_over_time:
    input:
        perf   = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl" for decoder in config["decoders"]],
        config = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/conf.yaml" for decoder in config["decoders"]],
    output:
        #f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{dataset_id}}_{{parcellation}}_{{feature}}_{'.'.join(config['aggr_conditions'])}_over_time.pdf",
        #f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{dataset_id}}_{{parcellation}}_{{feature}}_{'.'.join(config['aggr_conditions'])}_over_time.pkl",
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/performance_over_time.pdf",
        #f"results/plots/{{dataset_id}}/{{subset_id}}/{{parcellation}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/performance_over_time.pkl",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{{parcellation}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/plot_performance_over_time.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_performance_matrix:
    input:
        perf   = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/decoder_perf_across_timepoints.pkl",
        config = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/conf.yaml",
    output:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_performance_matrix.pdf",
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/performance_matrix.pdf",
        cluster = f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/performance_clustered.pdf",
        model = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}/performance_matrix_model.pkl",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_plot_matrix_performance.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_matrix.py"

rule plot_glassbrain:
    input:
        parcellation      = f"results/data/{{dataset_id}}/{{parcellation}}/{{dataset_id}}/data.h5",
        original_features = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Features/{cond}/{{feature}}/features.h5" for cond in config['aggr_conditions']],
        features          = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/best_feats.{config['export_type']}",

    output:
        plot              = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/circle_plot.pdf",
        interactive_plot  = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/glassbrain.html",
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/rfe/{'.'.join(config['aggr_conditions'])}/{{rfe_n}}/{{feature}}/plot_glassbrain.log",
    conda:
        "../envs/environment.yaml"
    resources:
        mem_mb=lambda wildcards, attempt: mem_res(wildcards,attempt,16000,16000)
    script:
        "../scripts/plot_glassbrain.py"



rule plot_performances_parcellations_over_time:
    input:
        perf = [f"results/data/{{dataset_id}}/{parcellation}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{decoder}/decoder_perf.pkl"
         for decoder in config["decoders"]
         for parcellation in config['parcellations']]
    output:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}_performance_over_time.pdf",
        #f"results/plots/{{dataset_id}}/{{subset_id}}/{{feature}}/{'.'.join(config['aggr_conditions'])}/performance_over_time_parcels.pkl",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        parcellations=list(config['parcellations']), #plot_feature_labels,
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{feature}}_performances_over_time.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_performances_features_over_time:
    input:
        perf = [f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{feature}/{decoder}/decoder_perf.pkl"
         for decoder in config["decoders"]
         for feature in config['features']]
    output:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}_performance_over_time.pdf",
        #f"results/plots/{{dataset_id}}/{{subset_id}}/{{parcellation}}/{'.'.join(config['aggr_conditions'])}/performance_over_time_features.pkl",
    params:
        conds=list(config['aggr_conditions']),
        decoders=config["decoders"],
        features=list(config['features']), #plot_feature_labels,
    log:
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{parcellation}}/{{parcellation}}_performances_over_time.log",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_performance_time.py"

rule plot_conf_matrix:
    input:
        conf_matrix = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/confusion_matrix.pkl",
        norm_conf_matrix = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/norm_confusion_matrix.pkl",
        class_labels = f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{decoder}}/class_labels.pkl",
    output:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_confusion_matrix.pdf",
        f"results/plots/{{dataset_id}}/{{subset_id}}/{'.'.join(config['aggr_conditions'])}/{{feature}}/{{parcellation}}/{{decoder}}/confusion_matrix.pdf",
    params:
        phases = config["phase_conditions"]
    log:
        f"results/data/{{dataset_id}}/{{parcellation}}/{{subset_id}}/Decoding/decoder/{'.'.join(config['aggr_conditions'])}/{{feature}}/plots/{{decoder}}_confusion_matrix.log",
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
