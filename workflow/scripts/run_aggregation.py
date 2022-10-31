import os
from snakemake.logging import logger

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log
from ci_lib import DecompData
import shutil
import time

#TODO move to utils or just remove whole script
def rec_iter_link(input,output):
    for i,o in zip(input,output):
        if isinstance(i,str) and isinstance(o,str):
            #os.symlink(src=str(i) , dst=str(o))
            #print(f"copy {r'{}'.format(i)} - > {r'{}'.format(o)}")
            print(Path(r'{}'.format(o)).parent)
            Path(r'{}'.format(o)).parent.touch()

            print(Path(r'{}'.format(o)))
            Path(r'{}'.format(o)).touch()


            shutil.copy2(Path(r'{}'.format(i)),Path(r'{}'.format(o)))

            time.sleep(1)

        elif hasattr(i, "__len__") and hasattr(o, "__len__") and not () :
            rec_iter_link(i,o)
        else:
            print("error")
            raise TypeError('Input/Output structure missmatched')

logger = start_log(snakemake)
if snakemake.config['limit_memory']:
    snakemake_tools.limit_memory(snakemake)
try:
    #snakemake_tools.save_conf(snakemake, sections=["parcellations"]) #fixed a bug as we dont apply parcellation to SVD and then prefilter fails to compare config as it won't contain parcellation
    timer_start = snakemake_tools.start_timer()
    print(snakemake.output)

    rec_iter_link(snakemake.input, snakemake.output)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
