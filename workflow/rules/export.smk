from snakemake_tools import calculate_memory_resource as mem_res

include: "common.smk"

rule export_data:
    input:
        lambda w: f"{DATA_DIR}/{unalias(w['decomp_alias'])}/{{parcellation}}/{unalias(w['dataset_alias'])}/data.h5"
    output:
        temporals = f"{EXPORTS_DIR}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/temporals.{{ext}}",
        spatials  = f"{EXPORTS_DIR}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/spatials.{{ext}}",
    log:
        f"{EXPORTS_DIR}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/export_data.{{ext}}.log"
    wildcard_constraints:
        decomp_alias  = r"[a-zA-Z\d_.#-]+",
        dataset_alias = r"[a-zA-Z\d_.#-]+",
        ext = r"(npy|npz|csv|pkl|mat)"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/export/export_data.py"

use rule export_data as export_data_deeper with:
    input:
        lambda w: f"{DATA_DIR}/{unalias(w['decomp_alias'])}/{{parcellation}}/{unalias(w['dataset_alias'])}/Features/{{rest}}/data.h5"
    output:
        temporals = f"{EXPORTS_DIR}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/{{rest}}/temporals.{{ext}}",
        spatials  = f"{EXPORTS_DIR}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/{{rest}}/spatials.{{ext}}",
    log:
        f"{EXPORTS_DIR}/{{decomp_alias}}/{{dataset_alias}}/{{parcellation}}/{{rest}}/export_data.{{ext}}.log"
