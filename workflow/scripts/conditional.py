# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"calciumimagingtools").absolute()))
from utils import snakemake_tools
# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry","parcellation","prefilters"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcellation","prefilters","conditions"])
timer_start = snakemake_tools.start_timer()

from data import DecompData

data = DecompData.load(snakemake.input[0])
data.conditions = snakemake.params[0]["trial_conditions"]

for cond, file in zip(data.conditions.keys(), snakemake.output[:-1]):
    phase = snakemake.params[0]["phase_conditions"][cond]
    if phase is None:
        start = None
        stop = None
    else:
        start = phase["start"] if "start" in phase else None
        stop = phase["stop"] if "stop" in phase else None
    data.conditions[cond, start:stop].save(file)

snakemake_tools.stop_timer(timer_start, f"{snakemake.rule}")
