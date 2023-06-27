'''
Contains various utility functions to be used in snakemake workflow scripts
'''

import datetime
import logging
import resource
import os
import pickle
import json

import yaml
import numpy

def load_wildcards(snakemake):
    '''
    Loads wildcards from a saved config yaml file
    Currently not used
    '''
    wildcards = []
    for path in snakemake.input["config"]:
        with open(path, "r", encoding='utf-8') as file:
            wildcards.append(yaml.safe_load(file)['wildcards'])


def save_conf(snakemake, sections, params=None, additional_config=None):
    '''
    Saves the given sections of the snakemake config and snakemake params into a yaml
    '''
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

def save(path, data):
    '''
    Saves `data` into `path` in a format specified by the extension of `path`
    '''
    if isinstance(data, numpy.ndarray):
        ext = path.split('.')[-1]
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

def load(path, dtype="float"):
    '''
    Loads data from a file saved into `path` in a format specified by the extension of `path`
    '''
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
    '''
    Matches the given sections of the snakemake config
    to a saved yaml file specified as snakemake input
    '''
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
    '''
    Checks if the given sections of the snakemake config match
    the config file specified as snakemake input
    '''
    if not match_conf(snakemake, sections=sections):
        if snakemake.config["different_config_inputs"] == 0:
            raise ValueError("Config used to generate input does not match current config!")
        if logger is not None:
            logger.warning("Config used to generate input does not match current config!")

def start_timer():
    '''Returns the current datetime, to be used with `stop_timer`'''
    return datetime.datetime.now()

def stop_timer(start, logger=None, silent=False):
    '''
    Takes a start datetime and returns the elapsed time,
    if not `silent` also logs this time delta
    '''
    delta = datetime.datetime.now() - start
    if not silent:
        (logging.getLogger(__name__) if logger is None else logger).info("Finished after %s", delta)
    return delta.total_seconds()

def limit_memory(snakemake, soft=True):
    '''
    Sets the python resource memory rlimit to the snakemake mem_mib resource limit
    '''
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)

    if soft:
        soft = snakemake.resources['mem_mib']*1024*1024
    else:
        hard = snakemake.resources['mem_mib']*1024*1024
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
