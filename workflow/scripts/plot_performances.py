import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools

logger = snakemake_tools.start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()


    decoders = [ dec.split('_')[0] for dec in snakemake.params['decoders']]
    conditions = snakemake.params['conds']

    features = ["\n".join(feat.split('_')[0:2]) if len(feat.split('_'))>3 else feat.split('_')[0] for feat in snakemake.params["features"]]

    #) for feat in snakemake.params["features"]


    #features = [ "\n".join(feat.split('_')) for feat in snakemake.params["features"]]

    dec_n = len(decoders)

    plt.figure(dpi=600)


    #Move to util
    def bold(text):
        return r"$\bf{" + text + "}$"
    #Move to util
    linebreak= '\n'
    subject_str = bold("Subjects(")+'#'+bold("Sessions)")+f": {', '.join(snakemake.params['subjects'])}"
    if 'parcellation' in snakemake.wildcards:
        title_str= bold("Parcellation")+f": {snakemake.wildcards['parcellation']}"
    else:
        title_str= bold("Feature")+f": {snakemake.wildcards['feature']}"

    conditions_str = bold("Conditions")+f":{linebreak}{linebreak.join(snakemake.params['conds'])}"
    trials_str = bold("Trials")+f":{linebreak}{linebreak.join([f'{k}: {v}'  for k,v in snakemake.params['trials'].items()])}"

    plt.gcf().text(0.78, 0.5, "\n \n".join([conditions_str,trials_str]) , fontsize=10, va='center')
    plt.subplots_adjust(right=0.75)



    plt.suptitle("\n".join([title_str,subject_str]),x=0.45)



    colors = cm.get_cmap('Accent',len(decoders))

    violin_plts= np.empty([len(features),len(decoders)],dtype='object')
    for f, feature in enumerate(features):
        if (f % 2) == 0:
            plt.axvspan(f-0.5, f+0.5, facecolor='0.2', alpha=0.05, zorder=-100)

        for d,decoder in enumerate(decoders):
            with open(snakemake.input[d+f*dec_n], "rb") as file:
                perf = pickle.load(file)
                violin_plts[f,d]=plots.colored_violinplot(perf, positions=f + np.arange(1) + ((d+1)*1/(dec_n+1))-0.5, widths=[1/(dec_n+2)], color=colors(d/dec_n))

    #plt.legend( [colors(i/len(decoders)) for i in range(len(decoders))], decoders )
    plt.legend( [ v['bodies'][0] for v in violin_plts[0]], decoders )
    plt.plot([-.5, len(features)-.5], [1/len(conditions), 1/len(conditions)], '--k')
    plt.text(len(features)-1,1/len(conditions)+0.01,'random chance',fontsize=7,ha='center')

    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(range(len(features)), features,fontsize=8)


    plt.savefig( snakemake.output[0] )

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
