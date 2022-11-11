from itertools import product as iterproduct

import json
import hashlib

def create_parameters( branch_conf, static_conf={} ):
    '''
    creates dictionary of unique names (used in paths) and corresponding parameters to be passed by snakemake
    '''
    default_params    = branch_conf["default"] if "default" in branch_conf else {}
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
    defaults = conditions["default"] if "default" in conditions else {}

    # create trial_conditions fitting for trial data by resolving column names and conditions from trial_conditions
    trial_condition     = config["trial_conditions"]
    trial_conditions    = { label :
                        {
                                trial_condition[condition]["column"] : [trial_condition[condition]["conds"][vals] for vals in value] if isinstance(value,list) else trial_condition[condition]["conds"][value]
                        for condition, value in (defaults | conds).items() if condition != "phase"}
                    for label, conds in conditions.items() if label != "default" }


    # lookup specified phases in phase_conditions
    phase_condition        = config["phase_conditions"]
    phase_conditions    = { label : phase_condition[(defaults | conds)["phase"]] if "phase" in (defaults | conds) else None
                    for label, conds in conditions.items() if label != "default" }
    return trial_conditions, phase_conditions, defaults

#TODO fix this mess / find a snakemake version, that fixes it
# taking input as an argument creates all kinds of bugs in snakemake...
#def calculate_memory_resource(wildcards, input, attempt, minimum=1000, step=1000, multiple=2):
def calculate_memory_resource(wildcards, attempt, minimum=1000, step=1000, multiple=2):
		#input_mb = input.size_mb
		#print(f"{wildcards}: {input_mb}\n\t{input}")
		#if input_mb == '<TBD>':
			#print("This should only appear in dry-run")
			#input_mb = 0
			#print(f"{wildcards}: {input_mb}\n\t{input}")
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