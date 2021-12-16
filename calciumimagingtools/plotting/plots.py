import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pathlib import Path
import scipy.io
import cv2

from bokeh.plotting import figure, output_file, show, from_networkx
from bokeh.models import (Label, LabelSet, HoverTool, BoxSelectTool,  TapTool, ColumnDataSource, LinearColorMapper,
                          Circle, EdgesAndLinkedNodes, NodesAndLinkedEdges, MultiLine)
from bokeh.palettes import Viridis, Viridis256, Spectral4

from shapely.geometry import Polygon

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from features import Feature_Type



def plot_glassbrain(DecompDataObject=None, frame=None,title='',dict_path=None,hl_areas=None,hl_edges=None):
    if dict_path is None:
        data_path = Path(__file__).parent.parent.parent/"resources"
        dict_path = data_path/"meta"/"anatomical.mat"
    areaDict = scipy.io.loadmat(dict_path ,simplify_cells=True)
    bitMasks = areaDict['areaMasks']
    labels = areaDict['areaLabels_wSide']

    polygons = []
    centers = np.empty((len(bitMasks),2))

    for c,mask in enumerate(bitMasks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Saves all contour points
        polygons.append(contours)

        centers[c,:] = Polygon(contours[0][:,0,:]).centroid.coords

        '''
        for i in contours:
            M = cv2.moments(i)
            if M['m00'] != 0:
                if(cx[c]>0): print(labels[c])
                cx[c] = (int(M['m10']/M['m00']))
                cy[c] = (int(M['m01']/M['m00']))
        '''

    #Weird nested list thats required for bokeh
    xs = [[[poly[0][:,:,0].flatten()]] for poly in polygons]
    ys = [[[poly[0][:,:,1].flatten()]] for poly in polygons]

    #Labels
    source = ColumnDataSource(data=dict(x=centers[:,0],y=centers[:,1],names=labels))
    labels = LabelSet(x='x',y='y', text='names', source=source,text_align='center',text_color='black')

    output_file("gfg.html")

    #Plotting
    mask_h , mask_w = bitMasks[0].shape
    frame_h, frame_w = frame.shape


    graph = figure(title = title,plot_width=frame_w*2, plot_height=mask_h*2 , y_range=[mask_h,0])

    # color values of the poloygons
    #color = ["red", "purple", "yellow"]

    # fill alpha values of the polygons
    fill_alpha = 0.1

    # plotting the graph and add labels
    colors= LinearColorMapper(palette=Viridis256,nan_color='white')


    graph.image(image=[np.flipud(frame[:,:mask_w])],x=0,y=frame_h,dw=frame_w,dh=frame_h,color_mapper=colors)
    graph.multi_polygons(xs, ys, line_color='black', line_width=2, line_alpha=0.5, fill_alpha=0.05)
    graph.add_layout(labels)

    #Create Graph
    if DecompDataObject is None:
        node_labels = dict()
        for i in range(64):  #N
            node_labels[i] = i+1
    else:
        node_labels = dict(enumerate(DecompDataObject.spatial_labels))

    graph.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())

    G=nx.karate_club_graph()
    G=nx.random_powerlaw_tree(len(centers),tries=10000)
    layout_nodes = dict(zip(range(len(centers)),centers))
    graph_renderer = from_networkx(G, layout_nodes, scale=1)

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    graph.renderers.append(graph_renderer)

    # displaying the model
    show(graph)



def colored_violinplot(*args, color=None, facecolor=None, edgecolor=None, **kwargs):
    violin_parts = plt.violinplot(*args, **kwargs)
    for part in violin_parts:
        # for pc in violin_parts['bodies']:
        parts = violin_parts[part] if part == 'bodies' else [violin_parts[part]]
        for pc in parts:
            if color is not None:
                pc.set_color(color)
            if facecolor is not None:
                pc.set_facecolor(facecolor)
            if edgecolor is not None:
                pc.set_edgecolor(edgecolor)
    return violin_parts


##Assumes that spatial is identical for all given temps
def plot_frame(temps, spatial, titles, plt_title):
    width = int(np.ceil(np.sqrt(len(temps))))
    height = int(np.ceil(len(temps) / width))
    #if height == 1: height = 2 #workaround for the moment as ax[h,w] wont work
    fig, ax = plt.subplots(height , width, constrained_layout=True, squeeze=False)
    fig.suptitle(plt_title)
    for h in range(height):
        for w in range(width):
            if h*width + w < len(temps):
                frame =  np.tensordot(temps[h*width + w], spatial, 1) #np.einsum( "n,nij->ij", temps[h*width + w], spatial) #np.tensordot(temps[w + h], spatial, (-1, 0)) #np.dot(spatial,temps[w*height + h]) #
                im = ax[h, w].imshow(frame)

                fig.colorbar(im, ax=ax[h, w])
                ax[h, w].set_title(titles[h*width + w])
                ax[h, w].set_xticks([])
                ax[h, w].set_yticks([])
                #plt.draw()
                #plt.pause(0.1)
    #plt.show()

    plt.savefig(plt_title, format='png')
    print("plotted")


def get_polygons():
    pass

def create_node_labels(DecompDataObject):
    pass

def get_nodes_pos(n_nodes,positions=None):
    pos = dict()
    if positions is None:
        for i in range(N):
            pos_circ[i] = np.array([np.sin(2*np.pi*(i/N+0.5/N)), np.cos(2*np.pi*(i/N+0.5/N))])
        return pos_circ
    else:
        pos = Positions

def get_edges():
    pass

def set_node_color():
    pass


def circle_rfe(selected_feats,DecompDataObject=None,title='RFE',n_nodes=None):
    pos_nodes = get_nodes_pos()


def graph_circle_plot(list_best_feat, n_nodes, title, feature_type, save_path=False,  node_labels=None):
    #%% network and plot properties
    N = n_nodes #20 # number of nodes
    print(N)
    # positions for circular layout with origin at bottoms
    pos_circ = dict()
    for i in range(N):
        pos_circ[i] = np.array([np.sin(2*np.pi*(i/N+0.5/N)), np.cos(2*np.pi*(i/N+0.5/N))])

    # channel labels need to be dict for nx
    if node_labels is None:
        node_labels = dict()
        for i in range(N):
            node_labels[i] = i+1
    else:
        node_labels = dict(enumerate(node_labels))

    # matrices to retrieve input/output channels from connections in support network
    mask = np.tri(N,N,0, dtype=bool) if feature_type == Feature_Type.UNDIRECTED else np.ones((N,N), dtype=bool)

    print(mask.shape)
    row_ind = np.repeat(np.arange(N).reshape([N,-1]),N,axis=1)
    col_ind = np.repeat(np.arange(N).reshape([-1,N]),N,axis=0)
    row_ind = row_ind[mask]
    col_ind = col_ind[mask]
    print(col_ind.shape)
    print(row_ind.shape)
    # plot RFE support network
    plt.figure(figsize=[10,10])
    plt.axes([0.05,0.05,0.95,0.95])
    plt.axis('off')
    plt.title=title
    if feature_type == Feature_Type.NODE: # nodal
        #list_best_feat = np.argsort(class_perfs.mean(0))[:n_edges] # select n best features
        node_color_aff = []
        g = nx.Graph()
        for i in range(N):
            g.add_node(i)
            if i in list_best_feat:
                node_color_aff += ['#71DFE7']
            else:
                node_color_aff += ['#E8F0F2']
        nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
        nx.draw_networkx_labels(g,pos=pos_circ,labels=node_labels)

    if feature_type == Feature_Type.DIRECTED or feature_type == Feature_Type.UNDIRECTED:
        #list_best_feat = np.argsort(class_perfs.mean(0))[:n_edges] # select n best features
        g = nx.Graph() if feature_type == Feature_Type.UNDIRECTED else nx.MultiDiGraph()
        for i in range(N):
            g.add_node(i)
        node_color_aff = ['#E8F0F2']*N
        list_ROI_from_to = [] # list of input/output ROIs involved in connections of support network
        for ij in list_best_feat:
            if(col_ind[ij] == row_ind[ij]): #checks for loops
                node_color_aff[col_ind[ij]]='#71DFE7' #colors node red
                print(ij)
            else:
                g.add_edge(col_ind[ij],row_ind[ij])
        print(g)
        nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
        nx.draw_networkx_labels(g,pos=pos_circ,labels=node_labels)
        nx.draw_networkx_edges(g,pos=pos_circ,edgelist=g.edges(),edge_color='#009DAE')
    if (not save_path):
        plt.show()
    else:
        plt.savefig(save_path)

    return g