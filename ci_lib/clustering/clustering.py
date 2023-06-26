import numpy as np

import logging
LOGGER = logging.getLogger(__name__)

import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.discriminant_analysis as skda
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
import sklearn.ensemble as skens
from sklearn.base import clone
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle
import pandas as pd

import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA

#Hide convergene warning for shuffled data
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

def cluster_UMAP():
    pass

def plot_DimRed(data, data_annot, discrete_color=False, color="t",symbol="phase",method="PCA", colorlist=None,cmap=None, comps = 3, path=None):
    if "PCA" in method:
            PCA_comps, total_var = PCA_(data,comps)
    elif "UMAP" in method:
            PCA_comps, total_var = UMAP_(data,comps)


    PCA_df = pd.DataFrame(PCA_comps)
    PCA_df=PCA_df.rename(columns={0: "PC1",1: "PC2",2:"PC3"})

    for annot,data in data_annot.items():
        PCA_df[annot] = data

    '''
    phases_discrete_cmap = {}
    phases = np.unique(data_phases)
    cmap = sns.color_palette("Set2" ,n_colors=len(phases)+1,as_cmap=True) if cmap is None else cmap
    for i,phase in enumerate(phases):
        phases_discrete_cmap[phase]= cmap((i+1)/(len(phases)+1))
    #if colorlist is None:
    #    #magic
    #    cmap=[cmap(l) for l in labels]
    LOGGER.info(PCA_df)
    
    labels = {str(i): f"PC {i+1}" for i in range(comps)}
    labels['color'] = 'Phase'

    fig = px.scatter_matrix(
        PCA_df,
        color=data_phases,
        dimensions=range(comps),
        labels=labels,
        title=f'Total Explained Variance: {total_var:.2f}%',
        #color_discrete_map = phases_discrete_cmap
    )
    #fig.update_traces(diagonal_visible=False)
    #fig.write_image(path)
    '''
    PCA_df = PCA_df.sort_values(color)

    def rgb_to_hex(r, g, b, a=1):
        return '#{:02x}{:02x}{:02x}'.format(int(255*r), int(255*g), int(255*b))

    if discrete_color:
        PCA_df = PCA_df.astype({color: str})
        colors_n = PCA_df[color].unique().size
        if "-1" in PCA_df[color].values:
            colors_n -= 1
            PCA_df = PCA_df.replace({color: "-1"},"unassigned")

        print(colors_n)
        color_list = sns.color_palette("tab20",20,as_cmap=True) #, 20)
        color_discrete_map = {str(i):rgb_to_hex(*color_list(i/20)) for i in range(colors_n)}
        color_discrete_map["unassigned"] = rgb_to_hex(*(0,0,0,0.5))
        print(color_discrete_map)

        fig = px.scatter_3d(PCA_df, x="PC1", y="PC2", z="PC3", color=color,color_discrete_map=color_discrete_map,symbol=symbol,title=f'Total Explained Variance: {total_var:.2f}%',hover_data=data_annot.keys())
    else:

        fig = px.scatter_3d(PCA_df, x="PC1", y="PC2", z="PC3", color=color,symbol=symbol,title=f'Total Explained Variance: {total_var:.2f}%',hover_data=data_annot.keys())

    # move colorbar
    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                                ticks="outside",
                                                tickprefix="Frame "))

    #h = [plt.plot([],[], color=cmap_phases((p+1)/(n_phases+1)) , marker="s", ms=i, ls="")[0] for p,_ in enumerate(phases.keys())]
    #ax.legend(handles=h, labels=list(phases.keys()), title="Trial Phase") #,loc=(-.27,.7),frameon=False)
    if "html" in path:
        fig.write_html(path)
    else:
        fig.write_image(path)

def PCA_(data, n_comps = 3):
    # data has shape observations x features


    #if not isinstance(data,pd.DataFrame):
    #    data = pd.DataFrame(data)

    pca = PCA(n_components=n_comps)
    comps = pca.fit_transform(data)

    total_var = pca.explained_variance_ratio_.sum() * 100

    return comps, total_var


    