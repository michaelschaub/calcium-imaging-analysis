from ci_lib.utils import snakemake_tools
from ci_lib import DecompData
from ci_lib.decomposition import anatomical_parcellation, fastICA

# redirect std_out to log file
snakemake_tools.redirect_to_log(snakemake)
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
