# add code library to path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.save_conf(snakemake, sections=["entry","parcelation","prefilters","conditions","feature_calculation"])

