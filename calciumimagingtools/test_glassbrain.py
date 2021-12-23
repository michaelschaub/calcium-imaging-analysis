import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)

from features import Raws,Means, Moup, Covariances, AutoCovariances, Feature_Type
from plotting import graph_circle_plot, plots, plot_glassbrain_bokeh,  construct_rfe_graph
from loading import load_task_data_as_pandas_df
from decomposition import anatomical_parcellation
from data import DecompData
import pickle as pkl


resl_path = Path(__file__).parent.parent/Path('results')
svd_path = str(resl_path/Path('GN06/SVD/data.h5'))
svd = DecompData.load(svd_path)
frame = Means.create(svd[:,30:75]).mean.pixel[0,:,:] #np.tensordot(Means.create(svd[:,30:75]),svd.spatials[:,:,:], 1)

#g = construct_rfe_graph(list_best_feat, comp,feature_data[feat][0].type,labels=ana.spatial_labels)
rfe_path = str(resl_path/Path('GN06/SVD/All/rfe/left_visual.left_tactile.left_vistact.right_visual.right_tactile.right_vistact/full/covariance_max_components~64/best_feats.pkl'))
with open(rfe_path, 'rb') as f:
    best_feats = pkl.load(f)

feattype=Feature_Type.UNDIRECTED
g = construct_rfe_graph(best_feats, 64,feattype)
frame=None
plot_glassbrain_bokeh(img=frame,graph=g,components_spatials=svd.spatials,components_labels=svd.spatial_labels)
