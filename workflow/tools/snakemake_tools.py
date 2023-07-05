from itertools import product as iterproduct

import json
import hashlib

import snakemake

def get_key_by_val(mydict,val):
    try:
        return list(mydict.keys())[list(mydict.values()).index(val)]
    except ValueError:
        raise ValueError("Value not found in dict") from None

def create_parameters( branch_conf, static_conf={} ):
    '''
    creates dictionary of unique names (used in paths) and corresponding parameters to be passed by snakemake
    '''
    default_params    = branch_conf.get("default", {})
    parameters    = {}
    for branch, br_params in branch_conf.items():


        if branch == "default":
            continue
        if br_params is None:
            br_params = {}
        # combine br_params with defaults into total params
        params = default_params | br_params

        optional = None
        if "optional" in params.keys():
            optional = params["optional"]
            del params["optional"]

        ### From here
        # create a pattern of form "branchname_{}_{}_...", with # of params replacement fields
        pattern    = "_".join( [f"{branch}"] +  ["{}"]*len(params) )
        # insert parameter names and parameter name replacement fields into pattern
        pattern    = pattern.format(*map( "{0}~{{{0}}}".format, params.keys() ))
        # create cartesian product over all parameter values
        values    = iterproduct( *params.values() )
        # create list parameter dictionaries from cartesian product
        values    = [ dict(zip(params.keys(), vals)) for vals in values]
        # include static parametrs if present
        if branch in static_conf and static_conf[branch] is not None:
            values = [ static_conf[branch] | vals for vals in values ]
        for vals in values:
            parameters[ pattern.format(**vals) ] = {"branch" : branch} | vals

        ###TODO same code twice

        ### From here
        if optional is not None:
            params = params | optional

            # create a pattern of form "branchname_{}_{}_...", with # of params replacement fields
            pattern    = "_".join( [f"{branch}"] +  ["{}"]*len(params) )
            # insert parameter names and parameter name replacement fields into pattern
            pattern    = pattern.format(*map( "{0}~{{{0}}}".format, params.keys() ))
            # create cartesian product over all parameter values
            values    = iterproduct( *params.values() )
            # create list parameter dictionaries from cartesian product
            values    = [ dict(zip(params.keys(), vals)) for vals in values]
            # include static parametrs if present
            if branch in static_conf and static_conf[branch] is not None:
                values = [ static_conf[branch] | vals for vals in values ]
            for vals in values:
                parameters[ pattern.format(**vals) ] = {"branch" : branch} | vals
        ### TODO this is just checking if it works  COPY PASTA

    return parameters

def create_conditions(conditions, config):
    defaults = conditions.get("default", {})

    # create trial_conditions fitting for trial data by resolving column names and conditions from trial_conditions
    trial_condition     = config["trial_conditions"]
    trial_conditions    = { label :
                        {
                                trial_condition[condition]["column"] : [trial_condition[condition]["conds"][vals] for vals in value] if isinstance(value,list) else trial_condition[condition]["conds"][value]
                        for condition, value in (defaults | conds).items() if condition != "phase"}
                    for label, conds in conditions.items() if label != "default" and "group" not in conds }

    # create list of conditions, that should be used for aggregration
    aggr_conditions = list(trial_conditions.keys())

    # lookup specified phases in phase_conditions
    phase_condition     = config["phase_conditions"]
    phase_conditions    = { label : phase_condition[(defaults | conds)["phase"]] if "phase" in (defaults | conds) else None
                    for label, conds in conditions.items() if label != "default" and "group" not in conds }

    # create group conditions
    group_conditions    = { label : list(conds["group"])
                    for label, conds in conditions.items() if "group" in conds }
    grouped_conditions  = [ l for conds in group_conditions.values() for l in conds ]
    # remove grouped conditions from list conditions to aggregrate
    aggr_conditions = [ label for label in aggr_conditions if label not in grouped_conditions ]
    # add new group conditions
    aggr_conditions.extend(group_conditions.keys())

    return aggr_conditions, trial_conditions, phase_conditions, group_conditions, defaults


def get_sessions(sets, content):
    sessions = content.get('subjects', {})
    for s in content.get('group',[]):
        s_sessions = get_sessions(sets, sets[s])
        sessions.update({ subj: (sess + sessions.get(subj,[])) for subj, sess in s_sessions.items() })
    return sessions

def flattened_sessions(sessions):
    sessions = [ (subj,sess) for subj, dates in sessions.items() for sess in dates ]
    sessions = sorted(set(sessions))
    return sessions

def dataset_path_hash(sessions, name, config):
    h = hashlib.md5()
    sessions = [f"{subj}.{date}" for subj, date in sessions]
    for s in sessions:
        h.update(bytes(s, "utf-8"))
    digest_lng = config.get('hash_digest_length', 8)
    digest = h.hexdigest()[:digest_lng]
    return '#'.join([name,digest])

def create_datasets(sets, config):
    '''
    Performs the conversion from the format used in the config to
    ´sets = { md5hashA : [(subj1,date1), ...], ...}´
    and ´aliases = { nameA: md5hashA, ... }´
    '''
    datasets = { name: flattened_sessions(get_sessions(sets, content)) for name, content in sets.items() }
    aliases = { name: dataset_path_hash(sessions, name, config) for name, sessions in datasets.items() }
    groups = { aliases[name]: [aliases[s] for s in content.get('group', [])] for name, content in sets.items() }

    datasets = {dataset_path_hash(sessions, name, config): sessions for name, sessions in datasets.items() }
    sessions = { id: ['-'.join(s) for s in sessions] for id, sessions in datasets.items()}
    return datasets, sessions, groups, aliases

def unalias_dataset(aliases, dataset):
    try:
        return aliases[dataset]
    except KeyError:
        return dataset

def alias_dataset(aliases, dataset):
    try:
        return get_key_by_val(aliases, dataset)
    except ValueError:
        return dataset

def temp_if_config(file, temp_config, rule=None):
    # check if rule specific config exists
    if rule in temp_config:
        temp = temp_config[rule]
    else:
        # get default value for all intermediates
        temp = temp_config.get("default", False)
    return snakemake.io.temp(file) if temp else file

#TODO fix this mess / find a snakemake version, that fixes it
# taking input as an argument creates all kinds of bugs in snakemake...
def calculate_memory_resource(wildcards, input, attempt, minimum=1000, step=1000, multiple=4):
    #input_mb = input.size_mb
    #if isinstance(input_mb, snakemake.common.TBDString):
        #input_mb = 0
    input_mb = 0
    return max(multiple*input_mb, minimum) + step*(attempt-1)

PARAMETER_EXPR = r"(_[a-zA-Z0-9'~.\[\], -]+)*"

def branch_match( branches, params=True ):
	return "(" + "|".join(branches) + ")" + (PARAMETER_EXPR if params else "")


def hash_config(config):

    return hashlib.md5(json.dumps(deep_stringize_dict_keys(config), sort_keys=True).encode('utf-8')).hexdigest() #Json.dump to force nested dicts to be sorted

def deep_stringize_dict_keys(item):
    """Converts all keys to strings in a nested dictionary"""

    #sets can't be serialized
    if isinstance(item,set):
        item = list(item)

    if isinstance(item, dict):
        return {str(k): deep_stringize_dict_keys(v) for k, v in item.items()}

    if isinstance(item, list):
        # This will check only ONE layer deep for nested dictionaries inside lists.
        # If you need deeper than that, you're probably doing something stupid.
        if any(isinstance(v, dict) for v in item):
            return [deep_stringize_dict_keys(v) if isinstance(v, dict) else v
                    for v in item]

    # I don't care about tuples, since they don't exist in JSON

    return item

def getKeys(dict):
    ''' Returns keys of dict. Failsafe for nested dicts where dict.keys() throws errors.'''
    list = []
    for key in dict.keys():
        list.append(key)
    return list
