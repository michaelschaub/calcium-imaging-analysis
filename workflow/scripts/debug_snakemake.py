# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"calciumimagingtools").absolute()))

from ci_lib.utils import snakemake_tools

# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
#snakemake_tools.save_conf(snakemake, sections=[])

print("snakemake:", dir(snakemake))
print("script_dir", snakemake.scriptdir)
print("input:", snakemake.input)
print("output:", snakemake.output)
print("log:", snakemake.log)
print("param:", snakemake.params)
print("config:", snakemake.config)
