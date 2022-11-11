import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import yaml

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools

logger = snakemake_tools.start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #  TODO this whole script is a mess -> encapsulate as function

    perf = []
    for path in snakemake.input["perf"]:
        with open(path, "rb") as f:
            perf.append(pickle.load(f))
    config = []
    for path in snakemake.input["config"]:
        with open(path, "r") as f:
            config.append(yaml.safe_load(f))

    decoders = [ dec.split('_')[0] for dec in snakemake.params['decoders']]
    conditions = snakemake.params['conds']

    fig=plt.figure()
    plt.suptitle("Decoding performance")
    violin_plts = []
    colors = cm.get_cmap('Accent',len(decoders)) #[np.arange(0,len(decoders),1)]

    for i,decoder in enumerate(decoders):
        flat_perfs =  np.array(perf[i]).flatten() #list(numpy.concatenate(perf[i]).flat) #Had dimension timepoints x reps
        violin_plts.append(plots.colored_violinplot(flat_perfs, positions=np.arange(1) + ((i+1)*1/(len(decoders)+1))-0.5, widths=[1/(len(decoders)+2)], color=colors(i/len(decoders))))


    plt.legend( [ v['bodies'][0] for v in violin_plts], decoders )
    plt.plot([-.5, 1-.5], [1/len(conditions), 1/len(conditions)], '--k') #1 cause we only have 1 feature for now
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(range(1), [snakemake.wildcards["feature"].split('_')[0]])

    ax = plt.gca()
    ax.set_ylim([0, 1])

    plt.savefig( snakemake.output[0] )


    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(fig, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
