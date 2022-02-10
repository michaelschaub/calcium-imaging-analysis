import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pathlib import Path
import scipy.io
import cv2


from shapely.geometry import Polygon, mapping

import plotly.express as px
from plotly.figure_factory import create_choropleth
import plotly.graph_objects as go

from ci_lib.plotting.plotly_extensions import addEdge
import networkx as nx
import geopandas

from scipy.ndimage import center_of_mass

#Interface
def plot_glassbrain_plotly(anatomical_file=None,img=None,graph=None,
                   components_spatials=None,components_labels=None,components_pos=None,
                   title=None):

    fig =  px.imshow(img,color_continuous_scale='viridis') #go.Figure()

    #Anatomical Layout
    for poly_trace in plt_polygons(fig,anatomical_file):
        fig.add_trace(poly_trace)

    #Labels
    #fig = plt_labels(fig,components_labels,components_pos)

    #Graph
    if components_pos is None:
        components_pos = calc_center(components_spatials)

    nodes, edges = plt_graph(fig,graph,components_pos)

    fig.add_trace(nodes)
    fig.add_trace(edges)



    fig.update_layout(height=1000, width=1000, showlegend=False, margin={"l":0,"r":0,"t":0,"b":0})
    fig.update_yaxes(autorange="reversed")

    fig.show()

def calc_center(spatials):
    spatials = np.nan_to_num(spatials)

    center = np.empty((len(spatials),2))
    for i,component in enumerate(spatials):
        center[i,:] = center_of_mass(np.absolute(component))
    return center

def plt_polygons(fig,path):
    geoJSON, labels = create_polygons(path)

    return [
            go.Scatter(
                **{
                    "x": [p[0] for p in f["geometry"]["coordinates"][0]],
                    "y": [p[1] for p in f["geometry"]["coordinates"][0]],
                    "fill": "toself",
                    "opacity":0.2,
                    "name": f["id"],
                },
                line=dict(color="black")
            )
            for f in geoJSON["properties"]
        ]

def create_polygons(path):
    #Fallback if no anatomical file is given
    if path is None:
        path = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"

    #Load anatomical masks and labels
    anatomical_file = scipy.io.loadmat(path,simplify_cells=True)
    areaMasks, areaLabels = anatomical_file['areaMasks'],anatomical_file['areaLabels_wSide']


    polygon_list = []
    #centers = np.empty((len(areaMasks),2))

    for c,mask in enumerate(areaMasks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Finds all contour points
        polygon_list.append(Polygon(contours[0][:,0,:]))
        #centers[c,:] = Polygon(contours[0][:,0,:]).centroid.coords

    return create_geoJSON(polygon_list,areaLabels), areaLabels
    #return polygons #, centers


def create_geoJSON(polygon_list,labels):
    geo_dict = {}
    geo_dict["type"] = "FeatureCollection"
    geo_dict["properties"] = [{"type": "Feature","id":labels[index], "geometry": a} for index,a in enumerate([mapping(poly) for poly in polygon_list])]
    return geo_dict

def plt_labels(fig, labels,positions):
    pass

#TODO Cite + Credit
def plt_graph(fig,graph,positions):
    # Controls for how the graph is drawn
    nodeColor = 'Orange'
    nodeSize = 15
    lineWidth = 5
    lineColor = 'gray'

    # Make a random graph using networkx
    G = graph
    pos = np.flip(positions, axis=1) #nx.layout.spring_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    # Make list of nodes for plotly
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    # Make a list of edges for plotly, including line segments that result in arrowheads
    edge_x = []
    edge_y = []
    for edge in G.edges():
        # addEdge(start, end, edge_x, edge_y, lengthFrac=1, arrowPos = None, arrowLength=0.025, arrowAngle = 30, dotSize=20)
        start = G.nodes[edge[0]]['pos']
        end = G.nodes[edge[1]]['pos']
        edge_x, edge_y = addEdge(start, end, edge_x, edge_y, .8, 'end', .04, 30, nodeSize)


    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=lineWidth, color=lineColor), hoverinfo='none', mode='lines')


    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(showscale=False, color = nodeColor, size=nodeSize))

    return  node_trace, edge_trace
    '''
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    '''
