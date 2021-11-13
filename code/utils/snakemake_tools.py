import sys
import datetime
import yaml

def redirect_to_log(snakemake):
    std_out = sys.stdout
    log_file = open(str(snakemake.log),'a')
    sys.stdout = log_file
    print(f"[{datetime.datetime.now()}] Log of rule {snakemake.rule}")
    return std_out

def save_conf(snakemake, sections, params=[], additional_config=None):
    config = {}
    for s in sections:
        config[s] = snakemake.config['rule_conf'][s]
    for p in params:
        config[p] = snakemake.params[p]
    config["wildcards"] = dict(snakemake.wildcards)
    if additional_config is not None:
        for key, item in additional_config:
            config[key] = item
    with open( snakemake.output["config"], 'w') as conf_file:
        yaml.dump(config, conf_file)

def match_conf(snakemake, sections):
    with open( snakemake.input["config"], 'r') as conf_file:
        config = yaml.safe_load(conf_file)
    for s in sections:
        if config[s] != snakemake.config["rule_conf"][s]:
            return False
    return True

def check_conf(snakemake, sections):
    if not match_conf(snakemake, sections=sections):
        if snakemake.config["different_config_inputs"] == 0:
            raise ValueError("Config used to generate input does not match current config!")
        else:
            warnings.warn("Config used to generate input does not match current config!")

