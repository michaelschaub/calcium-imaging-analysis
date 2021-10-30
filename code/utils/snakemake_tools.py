import sys
import datetime
import yaml

def redirect_to_log(snakemake):
    std_out = sys.stdout
    log_file = open(str(snakemake.log),'a')
    sys.stdout = log_file
    print(f"[{datetime.datetime.now()}] Log of rule {snakemake.rule}")
    return std_out

def save_conf(snakemake, sections, additional_config=None):
    config = {}
    for s in sections:
        config[s] = snakemake.config['rule_conf'][s]
    config["wildcards"] = dict(snakemake.wildcards)
    if additional_config is not None:
        for key, item in additional_config:
            config[key] = item
    with open( snakemake.output["config"], 'w') as conf_file:
        yaml.dump(config, conf_file)
