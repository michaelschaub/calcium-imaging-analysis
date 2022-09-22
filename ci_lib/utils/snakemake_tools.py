import sys
import datetime
import yaml
import logging
#import resource
import json

import os
import numpy
import pickle


def redirect_to_log(snakemake):
    # deprecated
    std_out = sys.stdout
    log_file = open(str(snakemake.log),'a')
    sys.stdout = log_file
    print(f"[{datetime.datetime.now()}] Log of rule {snakemake.rule}")
    return std_out

LOGLEVELS = {
            "DEBUG":logging.DEBUG,
            "INFO":logging.INFO,
            "WARNING":logging.WARNING,
            "ERROR":logging.ERROR,
            "CRITICAL":logging.CRITICAL, }

def start_log(snakemake):
    # logging <3.9 does not support encoding
    if sys.version_info[0] == 3 and sys.version_info[1] < 9 :
        logging.basicConfig(filename=str(snakemake.log), style="{",
                format="{asctime} {name}: {levelname}: {message}", datefmt="%b %d %H:%M:%S",
                level=LOGLEVELS[snakemake.config["loglevel"]])
    else:
        logging.basicConfig(filename=str(snakemake.log), encoding='utf-8', style="{",
                format="{asctime} {name}: {levelname}: {message}", datefmt="%b %d %H:%M:%S",
                level=LOGLEVELS[snakemake.config["loglevel"]])
    logger = logging.getLogger(f"{snakemake.rule}")
    logger.info(f"Start of rule")
    logger.info(f"Loglevel: {logger.getEffectiveLevel()}")
    return logger

def load_wildcards(snakemake):
    wildcards = []
    for path in snakemake.input["config"]:
        with open(path, "r") as f:
            wildcards.append(yaml.safe_load(f)['wildcards'])


def save_conf(snakemake, sections, params=[], additional_config=None):
    config = { 'static_params' : {}, 'branch_opts' : {} }
    for s in sections:
        config['branch_opts'][s] = json.loads(json.dumps(snakemake.config['branch_opts'][s]))
        if s not in ['conditions']:
            config['static_params'][s] = json.loads(json.dumps(snakemake.config['static_params'][s]))
    for p in params:
        config[p] = snakemake.params[p]
    config["wildcards"] = json.loads(json.dumps(dict(snakemake.wildcards)))
    if additional_config is not None:
        for key, item in additional_config:
            config[key] = item
    with open( snakemake.output["config"], 'w') as conf_file:
        yaml.dump(config, conf_file)

def save(snakemake, path, data):
    if isinstance(data, numpy.ndarray):
        match snakemake.config['export_type']:
            case 'csv':
                numpy.savetxt(path, data, delimiter=',')
            case 'npy':
                numpy.save(path, data)
            case 'npz':
                numpy.savez_compressed(path, data)
            case _:
                pass
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)

def load(snakemake, path, dtype="float"):
    _ , file_extension = os.path.splitext(path)
    match file_extension:
        case '.csv':
            return numpy.loadtxt(path, delimiter=',',dtype=dtype)
        case '.npy' | '.npz' :
            return numpy.load(path)
        case '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        case _:
            pass

def match_conf(snakemake, sections):
    with open( snakemake.input["config"], 'r') as conf_file:
        config = yaml.safe_load(conf_file)
    for s in sections:
        if s in config["static_params"] and config["static_params"][s] != json.loads(json.dumps(snakemake.config["static_params"][s])):
            return False
        if s in config["branch_opts"] and config["branch_opts"][s] != json.loads(json.dumps(snakemake.config["branch_opts"][s])):
            return False
    return True

def check_conf(snakemake, sections, logger=None):
    if not match_conf(snakemake, sections=sections):
        if snakemake.config["different_config_inputs"] == 0:
            raise ValueError("Config used to generate input does not match current config!")
        else:
            logger.warn("Config used to generate input does not match current config!") if logger is not None else None

def start_timer():
    return datetime.datetime.now()

def stop_timer(start, logger=None):
    delta = datetime.datetime.now() - start
    (logging.getLogger(__name__) if logger is None else logger).info(f"Finished after {delta}")

def limit_memory(snakemake, soft=True):
    '''
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)


    if soft:
        soft = snakemake.resources['mem_mb']*1024*1024
    else:
        hard = snakemake.resources['mem_mb']*1024*1024
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    '''
    pass