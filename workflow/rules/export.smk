from snakemake_tools import calculate_memory_resource as mem_res, unalias_dataset

unalias = lambda dataset: unalias_dataset(config["dataset_aliases"], dataset)

rule export_data:
    input:
        lambda w: f"results/data/{unalias(w['decomp_alias'])}/{{parcellation}}/{unalias(w['dataset_alias'])}/data.h5"
    output:
        temporals = f"results/exports/{config['name']}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/temporals.{{ext}}",
        spatials  = f"results/exports/{config['name']}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/spatials.{{ext}}",
    log:
        f"results/exports/{config['name']}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/export_data.{{ext}}.log"
    wildcard_constraints:
        #decomp_alias = r"[a-zA-Z\d_.#-]+",
        #dataset_alias = r"[a-zA-Z\d_.#-]+",
        ext = r"(npy|npz|csv|pkl)"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/export/export_data.py"

use rule export_data as export_data_deeper with:
    input:
        lambda w: f"results/data/{unalias(w['decomp_alias'])}/{{parcellation}}/{unalias(w['dataset_alias'])}/Features/{{rest}}/data.h5"
    output:
        temporals = f"results/exports/{config['name']}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/{{rest}}/temporals.{{ext}}",
        spatials  = f"results/exports/{config['name']}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/{{rest}}/spatials.{{ext}}",
    log:
        f"results/exports/{config['name']}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/{{rest}}/export_data.{{ext}}.log"
