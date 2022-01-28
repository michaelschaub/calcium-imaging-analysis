import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from ci_lib.plotting import plots
from ci_lib.utils import snakemake_tools

logger = snakemake_tools.start_log(snakemake)
timer_start = snakemake_tools.start_timer()


decoders = [ dec.split('_')[0] for dec in snakemake.params['decoders']]
conditions = snakemake.params['conds']
features = [ feat.split('_')[0] for feat in snakemake.params["features"]]

dec_n = len(decoders)

plt.figure()
plt.suptitle("_".join(snakemake.wildcards))
colors = cm.get_cmap('Accent',len(decoders))

violin_plts= np.empty([len(features),len(decoders)],dtype='object')
for f, feature in enumerate(features):
    for d,decoder in enumerate(decoders):
        with open(snakemake.input[d+f*dec_n], "rb") as file:
            perf = pickle.load(file)
            violin_plts[f,d]=plots.colored_violinplot(perf, positions=f + np.arange(1) + ((d+1)*1/(dec_n+1))-0.5, widths=[1/(dec_n+2)], color=colors(d/dec_n))

#plt.legend( [colors(i/len(decoders)) for i in range(len(decoders))], decoders )
plt.legend( [ v['bodies'][0] for v in violin_plts[0]], decoders )
plt.plot([-.5, len(features)-.5], [1/len(conditions), 1/len(conditions)], '--k') #1 cause we only have 1 feature for now
plt.yticks(np.arange(0, 1, 0.1))
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(range(len(features)), features)


plt.savefig( snakemake.output[0] )

snakemake_tools.stop_timer(timer_start, f"{snakemake.rule}")
