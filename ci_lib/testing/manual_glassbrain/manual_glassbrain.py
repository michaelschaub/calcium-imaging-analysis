from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent.parent).absolute()))

from ci_lib.utils import snakemake_tools
from ci_lib.features import Features, Means, Raws, Covariances, AutoCovariances, Moup, AutoCorrelations, Feature_Type
from ci_lib.plotting import graph_circle_plot, plot_glassbrain_bokeh
from ci_lib import DecompData
from ci_lib.rfe import RFE_pipeline, construct_rfe_graph

import numpy as np
import networkx as nx

n_comps = 64
list_best_feat = list(range(7*n_comps,8*n_comps))

resl_path = Path(__file__).parent.parent.parent.parent/'results'
svd_path = Path(resl_path/'GN06_2021-01-20_10-15-16/anatomical_ROI~[]/data.h5')
parcellation = DecompData.load(svd_path)
print(parcellation.spatial_labels[7])
feat_path= Path(resl_path/'GN06_2021-01-20_10-15-16/anatomical_ROI~[]/All/Features/left_vistact/moup_timelags~1_max_components~64/feature_data.h5')
feat = Moup.load(feat_path)
feat_val = feat._mou_ests[1].get_J()[7,:]

cutted_labels=parcellation.spatial_labels[:n_comps] if parcellation.spatial_labels is not None else None

feat_val = np.clip(feat_val,a_min=0,a_max=0.02)
feat_val = feat_val * 50
#feat_padded = np.zeros((n_comps,n_comps))
#feat_padded[list_best_feat]=feat_val

#print(feat_padded)
rfe_graph = construct_rfe_graph(list_best_feat, n_nodes = n_comps, feat_type = Feature_Type.DIRECTED, edge_weight= feat_val)
#nx.set_edge_attributes(rfe_graph, edge_attrs, "edge_color")
plot_glassbrain_bokeh(graph=rfe_graph,components_spatials=parcellation.spatials,components_labels=cutted_labels,save_path=resl_path/"glassbrain.html")
