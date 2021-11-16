# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"code").absolute()))
from utils import snakemake_tools
from data import DecompData
from decomposition import anatomical_parcellation


# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcelation"])
start = snakemake_tools.start_timer()



def anatom():
    svd = DecompData.load(snakemake.input[0])
    anatomical = anatomical_parcellation(svd)
    anatomical.save(snakemake.output[0])

def locaNMF():
    pass


parcelation = {'anatomical': anatom,
               'locaNMF': locaNMF}


parcelation[snakemake.wildcards['parcelation']]()

# only supported in 3.10
#match snakemake.wildcards['parcelation']:
#    case "anatomical":

snakemake_tools.stop_timer(start, f"{snakemake.rule}")