# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"calciumimagingtools").absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry","parcelation","prefilters"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcelation","prefilters","conditions"])
start = snakemake_tools.start_timer()

from data import DecompData

conditions = snakemake.params["conditions"]
data = DecompData.load(snakemake.input[0])
data.conditions = conditions

for cond, file in zip(data.conditions.keys(), snakemake.output[:-1]):
    data.conditions[cond].save(file)

snakemake_tools.stop_timer(start, f"{snakemake.rule}")
