#from nilearn import plotting
from pathlib import Path
import numpy as np
import scipy.io
import cv2

#Areas
data_path = Path(__file__).parent.parent.parent/"resources"
mask_path = data_path/"meta"/"areaMasks.mat"
areaMasks = scipy.io.loadmat(mask_path ,simplify_cells=True)
bitMasks = np.moveaxis(np.asarray(areaMasks['areaMasks']),-1,0)

#
polygons = []
for mask in bitMasks:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Saves all contour points
    polygons.append(contours)

#adjacency_matrix =   # n * n
#node_coords =       # n * 3

print("")
#plotting.plot_connectome(adjacency_matrix, node_coords, node_color='auto', node_size=50, edge_cmap=<matplotlib.colors.LinearSegmentedColormap object>, edge_vmin=None, edge_vmax=None, edge_threshold=None, output_file=None, display_mode='ortho', figure=None, axes=None, title=None, annotate=True, black_bg=False, alpha=0.7, edge_kwargs=None, node_kwargs=None, colorbar=False)[source]