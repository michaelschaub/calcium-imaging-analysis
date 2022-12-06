import os
from snakemake.logging import logger

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils.logging import start_log
from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
import shutil
import time

import itertools
from collections import defaultdict



def set_nested_dict_recursive(dict,keys,val=None):
    #adds the val to the dict nested under the list of keys
    print(keys)
    if len(keys) == 1:
        dict[keys[0]] = val
        return dict
    else:
        key = keys[0]
        dict[key] = dict.get(key, {}) #sets empty dict under key if it doesn't exist
        keys.pop(0) #removes that key from key list
        dict[key] = set_nested_dict_recursive(dict[key],keys,val) #continues the recursion on the next key
        return dict

def reorder_feats(generic_dict):
    #Due to the generic nature of the pipeline, the loaded feature mirror the internal file structure resulting from the independent runs
    #This structure is sessions x parcellation x selected_trials x conditions x features
    #Here we transform them to Parcellation x Feature x Condition x  (All Sessions + Individual Sessions)
    feats = {}

    for session,sdict in generic_dict.items():
        for parcellation,pdict in sdict.items():
            feats[parcellation]=feats.get(parcellation, {})
            for trials,tdict in pdict.items():
                for condition,cdict in tdict.items():

                    for feature, data in cdict.items():
                        feats[parcellation][feature] = feats[parcellation].get(feature, {})
                        feats[parcellation][feature][condition] = feats[parcellation][feature].get(condition, {})

                        feats[parcellation][feature][condition][session]= data 
    return feats

logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #Custom dict class that has nested dict as the default value if key does not exist
    nested_dict = lambda: defaultdict(nested_dict)
    nest = defaultdict(nested_dict)

    # loops over all arrays in iter in occuring order [X,Y] -> (x1,y1),(x1,y2), ...
    for i, keys in enumerate(itertools.product(*snakemake.params['iter'])):
        val = snakemake_tools.load(snakemake,snakemake.input[i]) 
        nest = set_nested_dict_recursive(nest, list(keys), val)

    #Removes lambda by converting to normal dict class
    generic_dict = dict(nest)

    #Reorders generic dict that mirrors pipeline structure into a more useful structure based on the output type
    if snakemake.params["reorder"] is not None:
        match snakemake.params["reorder"]:
            case 'feature':
                output_dict = reorder_feats(generic_dict)
            case other:
                output_dict = generic_dict
    #print(generic_dict)
 
    #print(output_dict)

    snakemake_tools.save(snakemake,snakemake.output["dict"],output_dict)

    #remove lambda exp. so nested_dict can be pickled
    #pkl_save_dict = json.loads(json.dumps(nest))
    #nest.default_factory = None
    # i just went with dict(nesteddict)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)