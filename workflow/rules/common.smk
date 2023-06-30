from snakemake_tools import alias_dataset, unalias_dataset, temp_if_config

def temp_c(in_file, rule=None):
    return temp_if_config(in_file, config.get("temporary_outputs",{}), rule)

alias   = lambda dataset:   alias_dataset(config["dataset_aliases"], dataset)
unalias = lambda dataset: unalias_dataset(config["dataset_aliases"], dataset)

DATA_DIR    = f"results/data"
PLOTS_DIR   = f"results/plots/{config['name']}"
EXPORTS_DIR = f"results/exports/{config['name']}"
