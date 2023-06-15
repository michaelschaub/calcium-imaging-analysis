import datetime
import logging
import resource
import os
import pickle
import json

import yaml
import numpy

def load_wildcards(snakemake):
    wildcards = []
    for path in snakemake.input["config"]:
        with open(path, "r", encoding='utf-8') as file:
            wildcards.append(yaml.safe_load(file)['wildcards'])


def save_conf(snakemake, sections, params=None, additional_config=None):
    if params is None:
        params = []
    config = { 'static_params' : {}, 'branch_opts' : {} }
    for sec in sections:
        config['branch_opts'][sec] = json.loads(json.dumps(
                                                snakemake.config['branch_opts'][sec]))
        if sec not in ['conditions']:
            config['static_params'][sec] = json.loads(json.dumps(
                                                snakemake.config['static_params'][sec]))
    for par in params:
        config[par] = snakemake.params[par]
    config["wildcards"] = json.loads(json.dumps(dict(snakemake.wildcards)))
    if additional_config is not None:
        for key, item in additional_config:
            config[key] = item
    with open( snakemake.output["config"], 'w', encoding='utf-8') as conf_file:
        yaml.dump(config, conf_file)

def save(snakemake, path, data):
    if isinstance(data, numpy.ndarray):
        ext = snakemake.config['export_type']
        if ext == 'csv':
            numpy.savetxt(path, data, delimiter=',')
        elif ext == 'npy':
            numpy.save(path, data)
        elif ext == 'npz':
            numpy.savez_compressed(path, data)
        else:
            pass
    else:
        with open(path, 'wb', encoding='utf-8') as file:
            pickle.dump(data, file)

def load(snakemake, path, dtype="float"):
    _ , file_extension = os.path.splitext(path)
    ext = file_extension
    if ext=='.csv':
        return numpy.loadtxt(path, delimiter=',',dtype=dtype)
    if ext in ('.npy', '.npz'):
        return numpy.load(path)
    if ext=='.pkl':
        with open(path, 'rb', encoding='utf-8') as file:
            data = pickle.load(file)
        return data
    raise ValueError("Unrecognised file extension")

def match_conf(snakemake, sections):
    with open( snakemake.input["config"], 'r', encoding='utf-8') as conf_file:
        config = yaml.safe_load(conf_file)
    for sec in sections:
        if (sec in config["static_params"]
            and config["static_params"][sec] != json.loads(json.dumps(
                                                snakemake.config["static_params"][sec]))):
            return False
        if (sec in config["branch_opts"]
            and config["branch_opts"][sec] != json.loads(json.dumps(
                                                snakemake.config["branch_opts"][sec]))):
            return False
    return True

def check_conf(snakemake, sections, logger=None):
    if not match_conf(snakemake, sections=sections):
        if snakemake.config["different_config_inputs"] == 0:
            raise ValueError("Config used to generate input does not match current config!")
        if logger is not None:
            logger.warning("Config used to generate input does not match current config!")

def start_timer():
    return datetime.datetime.now()

def stop_timer(start, logger=None, silent=False):
    delta = datetime.datetime.now() - start
    if not silent:
        (logging.getLogger(__name__) if logger is None else logger).info("Finished after %s", delta)
    return delta.total_seconds()

def limit_memory(snakemake, soft=True):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)


    if soft:
        soft = snakemake.resources['mem_mib']*1024*1024
    else:
        hard = snakemake.resources['mem_mib']*1024*1024
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
