import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import yaml

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()
    with open(snakemake.input["conf_matrix"], "rb") as f:
        confusion_matrices = np.asarray(pickle.load(f))

    with open(snakemake.input["norm_conf_matrix"], "rb") as f:
        norm_confusion_matrices = np.asarray(pickle.load(f))

    with open(snakemake.input["class_labels"], "rb") as f:
        class_labels = np.asarray(pickle.load(f))

    sns.set(rc={'figure.figsize':(4 , 3.7)})
    # t x class x class
    conf_matrix = np.mean(confusion_matrices,axis=0) #mean over time #TODO some time-resolved presentation
    conf_matrix = np.sum(conf_matrix,axis=0).astype("int") #sum over runs

    norm_conf_matrix = np.mean(norm_confusion_matrices,axis=0) #mean over time #TODO some time-resolved presentation
    norm_conf_matrix = np.mean(norm_conf_matrix,axis=0) #mean over runs

    fig = sns.heatmap(norm_conf_matrix, cmap=plt.cm.Blues,square=True, vmin=0, vmax=1, annot=conf_matrix, xticklabels=class_labels, yticklabels=class_labels, fmt='.0f',cbar_kws={"shrink": .8})

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    fig.tick_params(left=True, bottom=True)

    feature = snakemake.wildcards["feature"]
    parcellation = snakemake.wildcards["parcellation"]
    fig.set_title(f"{parcellation}_{feature}",pad=10)

    plt.tight_layout()
    fig.figure.savefig( snakemake.output[0] )
    fig.figure.savefig( snakemake.output[1] )

    #with open(snakemake.output[0], 'wb') as f:
    #    pickle.dump(fig.figure, f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
