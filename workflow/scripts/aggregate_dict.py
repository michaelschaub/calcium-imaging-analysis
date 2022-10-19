import os
from snakemake.logging import logger

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
import shutil
import time

import itertools
from collections import defaultdict



def set_nested_dict_recursive(dict,keys,val=None):
    print(keys)
    if len(keys) == 1:
        print(keys)
        dict[keys[0]] = val
        return dict
    else:
        key = keys[0]
        dict[key] = dict.get(key, {}) #sets empty dict under key if it doesn't exist
        keys.pop(0) #removes that key from key list
        dict[key] = set_nested_dict_recursive(dict[key],keys,val) #continues the recursion on the next key
        return dict

logger = snakemake_tools.start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    #snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()

    nested_dict = lambda: defaultdict(nested_dict)
    nest = defaultdict(nested_dict)

    # loops over all arrays in iter in occuring order [X,Y] -> (x1,y1),(x1,y2), ...
    for i, keys in enumerate(itertools.product(*snakemake.params['iter'])):
        print(keys)
        val = snakemake_tools.load(snakemake,snakemake.input[i]) #TODO this assumes that the order of inputs is unchanged
        nest = set_nested_dict_recursive(nest, list(keys), val)

    #remove lambda exp. so nested_dict can be pickled
    #pkl_save_dict = json.loads(json.dumps(nest))
    #nest.default_factory = None
    # i just went with dict(nesteddict)

    snakemake_tools.save(snakemake,snakemake.output["dict"],dict(nest))

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)