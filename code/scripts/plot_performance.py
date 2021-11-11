import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from plotting import plots



perf = []
for path in snakemake.input:
    with open(path, "rb") as f:
        perf.append(pickle.load(f))

decoders = snakemake.params['decoders']
conditions = snakemake.params['conds']

fig=plt.figure()
plt.suptitle("Decoding performance")
violin_plts = []
colors = cm.get_cmap('Accent',len(decoders)) #[np.arange(0,len(decoders),1)]

for i,decoder in enumerate(decoders):
    violin_plts.append(plots.colored_violinplot(perf[i], positions=np.arange(1) + ((i+1)*1/(len(decoders)+1))-0.5, widths=[1/(len(decoders)+2)], color=colors(i/len(decoders))))


plt.legend( [ v['bodies'][0] for v in violin_plts], decoders )
plt.plot([-.5, 1-.5], [1/len(conditions), 1/len(conditions)], '--k') #1 cause we only have 1 feature for now
plt.yticks(np.arange(0, 1, 0.1))
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(range(1), [snakemake.wildcards["feature"]])


plt.savefig( snakemake.output[0] )


with open(snakemake.output[1], 'wb') as f:
    pickle.dump(fig, f)
