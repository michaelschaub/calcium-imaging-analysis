# add code library to path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry","parcelation"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcelation","prefilters"])

import shutil

if snakemake.wildcards['filter'] == 'All':
    shutil.copyfile( snakemake.input[0], snakemake.output[0] )
else:
    raise ValueError(f"Filter f{snakemake.wildcards['filter']} not recognized")
print(f"Applied filter {snakemake.wildcards['filter']}")
