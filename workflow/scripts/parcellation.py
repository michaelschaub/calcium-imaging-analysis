# add code library to path
from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent/"calciumimagingtools").absolute()))
from utils import snakemake_tools
from data import DecompData
from decomposition import anatomical_parcellation, fastICA

# redirect std_out to log file
logger = snakemake_tools.start_log(snakemake)
snakemake_tools.check_conf(snakemake, sections=["entry"])
snakemake_tools.save_conf(snakemake, sections=["entry","parcellation"])
start = snakemake_tools.start_timer()

def anatom():
    svd = DecompData.load(snakemake.input[0])
    anatomical = anatomical_parcellation(svd, dict_path=snakemake.input["meta"])
    anatomical.save(snakemake.output[0])

def locaNMF():
    pass

def ICA():
    svd = DecompData.load(snakemake.input[0])
    ica = fastICA(svd, 64) #snakemake.config
    ica.save(snakemake.output[0])


parcellation = {'anatomical': anatom,
                'ICA':ICA,
               'locaNMF': locaNMF}
parcellation[snakemake.wildcards['parcellation']]()

snakemake_tools.stop_timer(start, f"{snakemake.rule}")
