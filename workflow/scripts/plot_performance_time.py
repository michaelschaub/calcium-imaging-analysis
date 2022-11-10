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

    
    fig=plt.figure(figsize=[20,10])
    plt.suptitle("Decoding performance")
    violin_plts = []
    colors = cm.get_cmap('Accent',len(decoders)) #[np.arange(0,len(decoders),1)]
    legend = []

    for i,decoder in enumerate(decoders):
        timesteps = len(perf[i])
        width = 0.8 #/(timesteps+2)

        for j,timestep_perf in enumerate(perf[i]):
            #flat_perfs =  np.array(perf[i]).flatten() #list(numpy.concatenate(perf[i]).flat) #Had dimension timepoints x reps
            pos = np.arange(1) + (j) #*1/(timesteps+1))-0.5
            violin_plts.append(plots.colored_violinplot(timestep_perf, positions= pos , widths=[width], color=colors(i/len(decoders))))
        
        legend.append(violin_plts[-1]['bodies'][0])

    plt.legend( legend, decoders,loc='lower left')
    plt.plot([-.5, timesteps -.5], [1/len(conditions), 1/len(conditions)], '--k')
    plt.text(timesteps-0.8,1/len(conditions)+0.01,'random chance',fontsize=7,ha='center')

   
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(np.arange(0, 1+timesteps, 7.5),np.arange(-30, 1+timesteps-30, 7.5)) #,np.arange(0, 1+timesteps, 7.5)*(1/15))
    plt.xlabel('Frames (15 Hertz)')

    ax = plt.gca()
    ax.set_ylim([0, 1.05])

    plt.savefig( snakemake.output[0] )


    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(fig, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
