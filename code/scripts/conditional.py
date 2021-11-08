# add code library to path
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry","parcelation","prefilters"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcelation","prefilters","conditions"])

from data import DecompData

conditions = snakemake.params["conditions"]
data = DecompData.load(snakemake.input[0])
data.conditions = conditions

for cond, file in zip(data.conditions.keys(), snakemake.output[:-1]):
    data.conditions[cond].save(file)
