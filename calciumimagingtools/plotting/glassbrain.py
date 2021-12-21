import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pathlib import Path
import scipy.io
import cv2

from bokeh.plotting import figure, output_file, save, show, from_networkx
from bokeh.models import (Label, LabelSet, HoverTool, BoxSelectTool,  TapTool, ColumnDataSource, LinearColorMapper,
                          Circle, EdgesAndLinkedNodes, NodesAndLinkedEdges, MultiLine, Patch,MultiPolygons,Patches)
from bokeh.palettes import Viridis, Viridis256, Spectral4

from shapely.geometry import Polygon

import geopandas


#Interface
def plt_glassbrain(anatomical_file=None,img=None,graph=None,components_spatials=None,components_labels=None,components_pos=None):

    #Needs w, h
    fig = create_fig(img,poly)

    fig = plt_frame(fig,img)

    fig = plt_polygons(fig,anatomical_file)

    fig = plt_labels(fig,components_labels,components_positions)

    fig = plt_graph(fig,graph,components_positions)

    pass



#Plotly
def create_fig(img=None,poly=None):

    return fig

def plt_frame(fig,img):
    pass

def plt_polygons(fig,path):



    pass

def create_polygons(path):
    #Fallback if no anatomical file is given
    if path is None:
        path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"

    #Load anatomical masks and labels
    anatomical_file = scipy.io.loadmat(path,simplify_cells=True)
    areaMasks, areaLabels = anatomical_file['areaMasks'],anatomical_file['areaLabels_wSide']


    polygons = []
    centers = np.empty((len(areaMasks),2))

    for c,mask in enumerate(areaMasks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Finds all contour points
        polygons.append(contours)
        centers[c,:] = Polygon(contours[0][:,0,:]).centroid.coords
    return polygons, centers


def create_geoJSON(polygons,labels):
    geo_dict["features"] = [{"type": "Feature", "geometry": a} for a in [geometry.mapping(b) for b in polygons]]
    my_geojson = json.dumps(geo_dict)

def plt_labels(fig, labels,positions):
    pass

def plt_graph(fig,graph,positions):
    pass

def plt_glassbrain_plotly():
    pass