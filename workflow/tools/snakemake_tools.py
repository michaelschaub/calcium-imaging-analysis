from itertools import product as iterproduct

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
    return parameters

def create_conditions(conditions):
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

