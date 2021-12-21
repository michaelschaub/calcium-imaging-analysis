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

import sys
from pathlib import Path
sys.path.append(Path(__file__).parent)
from features import Feature_Type

#Interface
def plt_glassbrain(graph=None, comp_pos=None, bg_img=None,meta_file=None,title=''):
    #Load fallback
    if meta_file is None:
        meta_file = Path(__file__).parent.parent.parent/"resources"/"meta"/"anatomical.mat"
    metaDict = scipy.io.loadmat(meta_file ,simplify_cells=True)
    areaMasks, areaLabels = metaDict['areaMasks'],metaDict['areaLabels_wSide']

    #Calc poly, center, labels,
    areaPolygons, areaCenters = calc_polygons(areaMasks)
    labelset = create_labelset(areaCenters,areaLabels)
    #graph = None
    #bg_img = None

    #Bounding Box
    poly_bbox = list(areaMasks[0].shape)
    img_bbox =list(bg_img.shape) if bg_img is not None else np.full((2), np.nan)
    y_range, x_range= tuple(np.maximum(poly_bbox, img_bbox))

    #Plot Glassbrain
    draw_glassbrain(areaPolygons,labelset, graph, bg_img, x_range=x_range, y_range=y_range, scale=2,title=title)

# Helper Functions
def construct_rfe_graph(selected_feats, n_nodes, feat_type, labels):
    '''
    if labels is None:
        node_labels = dict()
        for i in range(n_nodes): node_labels[i] = i+1
    else:
        node_labels = dict(enumerate(labels))
    '''

    # matrices to retrieve input/output channels from connections in support network
    mask = np.tri(n_nodes,n_nodes,0, dtype=bool) if feat_type == Feature_Type.UNDIRECTED else np.ones((n_nodes,n_nodes), dtype=bool)
    row_ind = np.repeat(np.arange(n_nodes).reshape([n_nodes,-1]),n_nodes,axis=1)
    col_ind = np.repeat(np.arange(n_nodes).reshape([-1,n_nodes]),n_nodes,axis=0)
    row_ind = row_ind[mask]
    col_ind = col_ind[mask]

    default_color= 'gray'
    selected_color= 'yellow'

    if feat_type == Feature_Type.NODE: # nodal
        g = nx.Graph()
        for i in range(n_nodes):
            g.add_node(i)
            g.nodes[i]['selected'] = (i in selected_feats)
            g.nodes[i]['color'] = selected_color if (i in selected_feats) else default_color
        #g.nodes[selected_feats]['color'] = selected_color if (i in selected_feats) else default_color

    if feat_type == Feature_Type.DIRECTED or feat_type == Feature_Type.UNDIRECTED:
        g = nx.Graph() if feat_type == Feature_Type.UNDIRECTED else nx.MultiDiGraph()
        for i in range(n_nodes):
            g.add_node(i)
            g.nodes[i]['selected'] = False
            g.nodes[i]['color'] = default_color

        for ij in selected_feats:
            if(col_ind[ij] == row_ind[ij]): #checks for loops
                g.nodes[col_ind[ij]]['selected'] = True
                g.nodes[col_ind[ij]]['color'] = selected_color #colors node red
            else:
                g.add_edge(col_ind[ij],row_ind[ij])

        #Remove nodes with degree 0
        remove = [node for node,degree in dict(g.degree()).items() if degree == 0]
        g.remove_nodes_from(remove)

    return g

def calc_polygons(masks):
    polygons = []
    centers = np.empty((len(masks),2))

    for c,mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #Finds all contour points
        polygons.append(contours)
        centers[c,:] = Polygon(contours[0][:,0,:]).centroid.coords

    return polygons, centers

def format_polygons(polys,labels):
    xs = [[[poly[0][:,:,0].flatten()]] for poly in polys]
    ys = [[[poly[0][:,:,1].flatten()]] for poly in polys]
    return ColumnDataSource(data=dict(xs=xs, ys=ys,name=labels))

def create_labelset(centers, labels):
    source = ColumnDataSource(data=dict(x=centers[:,0],y=centers[:,1],names=labels))
    return LabelSet(x='x',y='y', text='names', source=source,text_align='center',text_color='black')


#Plotly
def create_plt(img=None,poly=None):

    return fig

def plt_frame(fig,img):
    pass

def plt_polygons(fig,poly):
    pass

def plt_labels(fig,positions, labels):
    pass

def plt_graph(fig,graph):
    pass

def plt_glassbrain_plotly():
    pass





def draw_glassbrain(polygons=None,labelset=None,graph=None,bg_img=None,x_range=None,y_range=None,scale=1,title=''):
    width, height = x_range * scale,y_range * scale
    fig = figure(title = title,plot_width=width, plot_height=height ,
                   x_range=[0,x_range], y_range=[y_range,0])

    #Background Image
    if bg_img is not None:
        colors= LinearColorMapper(palette=Viridis256,nan_color=(0,0,0,0))
        img_h, img_w = bg_img.shape
        fig.image(image=[np.flipud(bg_img)],x=0,y=img_h,dw=img_w,dh=img_h,color_mapper=colors)

    #Polygons with Labels
    if None not in (polygons,labelset):
        poly_glyph = MultiPolygons(xs="xs", ys="ys", line_color='black', line_width=2, line_alpha=0.5, fill_alpha=0.05)
        poly_render = fig.add_glyph(format_polygons(polygons,labelset.source.data["names"]), poly_glyph)
        #label_render = fig.add_layout(labelset)

        fig.add_tools(HoverTool(renderers=[poly_render],point_policy = 'follow_mouse',tooltips="@name"), TapTool(), BoxSelectTool())

    #Graph
    if graph is not None:
        #Currently only center for anatomical
        nodes = list(graph.nodes)
        print(nodes)
        centers = list(zip(labelset.source.data["x"][nodes] , labelset.source.data["y"][nodes]))
        layout_nodes = dict(zip(nodes,centers))

        graph_renderer = from_networkx(graph, layout_nodes, node_attrs=['color','selected'],scale=1)

        #xs=[[poly[0][:,:,0].flatten()]for poly in polygons ]
        #ys=[[poly[0][:,:,1].flatten()]for poly in polygons ]


        #graph_renderer.node_renderer.data_source.add(xs, 'xs')
        #graph_renderer.node_renderer.data_source.add(ys, 'ys')

        #color = nx.get_node_attributes(graph,'color')
        #graph_renderer.node_renderer.data_source.add(list(color.values()),'color')

        selected_dict = nx.get_node_attributes(graph,'selected')
        selected_nodes = [k for k,v in selected_dict.items() if v] #np.asarray((nx.get_node_attributes(graph,'selected')).values(),dtype=bool)
        alpha_nodes = np.full(64,0.5) #TODO Hardcoded number of comps
        alpha_nodes[selected_nodes] = 1
        graph_renderer.node_renderer.data_source.add(alpha_nodes[nodes],'alpha_nodes')

        #graph_renderer.node_renderer.data_source.add(list(color.values()),'color')

        graph_renderer.node_renderer.data_source.add(labelset.source.data["names"][nodes],"label")
        '''
        class Custom_Patch(Patch):
            __view_model__ = 'Patch'
            __subtype__ = 'Custom_Patch'
            __view_module__ = '__main__'

            def __init__(self,*args,xs=None,ys=None,**kwargs):
                kwargs['x']=xs #kwargs['xs']
                kwargs['y']=ys #kwargs['ys']
                #self.x = xs
                #self.y = ys

                if args is not None:
                    if len(args)>=2:
                       kwargs['x']=args[0] #kwargs['xs']
                       kwargs['y']=args[1] #kwargs['ys']

                super().__init__(**kwargs)
        '''

        #graph_renderer.node_renderer.glyph = Custom_Patch(xs='xs',ys='ys')
        graph_renderer.node_renderer.glyph = Circle(size=20, fill_color="color", fill_alpha="alpha_nodes") #Patch
        graph_renderer.node_renderer.selection_glyph = Circle(size=20, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=20, fill_color=Spectral4[1])


        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        #fig.add_tools(HoverTool(renderers=[graph_renderer.node_renderer],tooltips=[("Area", "@label"), ("x","@xs"),("y","@ys")]), TapTool(), BoxSelectTool())
        fig.add_tools(HoverTool(renderers=[graph_renderer.edge_renderer],tooltips=None),
                      HoverTool(renderers=[graph_renderer.node_renderer],tooltips=[("Component","@label")]),
                      TapTool(renderers=[graph_renderer.edge_renderer,graph_renderer.node_renderer]),
                      BoxSelectTool())


        graph_renderer.selection_policy = NodesAndLinkedEdges()
        graph_renderer.inspection_policy = EdgesAndLinkedNodes()

        fig.renderers.append(graph_renderer)

    # displaying the model


    output_file(filename=f"{title}.html", title=title)
    save(fig)

    show(fig)

def plot_glassbrain(DecompDataObject=None, frame=None,connectivity_graph=None, title='',dict_path=None,hl_areas=None,hl_edges=None):
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
    label_set = LabelSet(x='x',y='y', text='names', source=source,text_align='center',text_color='black')

    output_file("gfg.html")

    #Plotting
    mask_h , mask_w = bitMasks[0].shape
    frame_h, frame_w = frame.shape


    graph = figure(title = title,plot_width=frame_w*2, plot_height=mask_h*2 , x_range=[0,np.maximum(mask_w,frame_w)], y_range=[np.maximum(mask_h,frame_h),0])

    # color values of the poloygons
    #color = ["red", "purple", "yellow"]

    # fill alpha values of the polygons
    fill_alpha = 0.1

    # plotting the graph and add labels
    colors= LinearColorMapper(palette=Viridis256,nan_color='white')


    graph.image(image=[np.flipud(frame[:,:mask_w])],x=0,y=frame_h,dw=frame_w,dh=frame_h,color_mapper=colors)
    graph.multi_polygons(xs, ys, line_color='black', line_width=2, line_alpha=0.5, fill_alpha=0.05)
    graph.add_layout(label_set)

    #Create Graph
    if DecompDataObject is None:
        node_labels = dict()
        for i in range(64):  #N
            node_labels[i] = i+1
    else:
        node_labels = dict(enumerate(DecompDataObject.spatial_labels))



    G=nx.karate_club_graph()

    #Give polynoms to nodes
    #source =
    #sources = [ ColumnDataSource(dict(x=poly[0][:,:,0].flatten() , y=poly[0][:,:,1].flatten() )) for poly in polygons]


    for v in G:
        G.nodes[v]['label']= labels[v]
        G.nodes[v]['xs'] = polygons[v][0][:,:,0].flatten()
        G.nodes[v]['ys'] =  polygons[v][0][:,:,1].flatten()


    #nx.set_node_attributes(G, name='xs_Poly', values=xs)
    #nx.set_node_attributes(G, name='ys_Poly', values=ys)
    #xs = [poly[0][:,:,0].flatten()for poly in polygons ]
    #nx.set_node_attributes(G, name='x', values=[poly[0][:,:,0].flatten()for poly in polygons ])
    #nx.set_node_attributes(G, name='y', values=[poly[0][:,:,1].flatten()for poly in polygons ])
    #nx.set_node_attributes(G,name'test',values=)

    #G=nx.random_powerlaw_tree(len(centers),tries=10000)

    #G=connectivity_graph


    layout_nodes = dict(zip(range(len(centers)),centers))
    #layout_nodes = lay_dict
    graph_renderer = from_networkx(G, layout_nodes, scale=1)

    '''
    class NodePatch(Patch):
        x = None
        y = None

        __init__(self, *args):

            Super().__init__(self, *args)
    '''

    graph_renderer.node_renderer.glyph = Circle(x='xs',y='ys',size=15, fill_color=Spectral4[0]) #Patch
    #graph_renderer.node_renderer.glyph = Patch(x='x',y='y')
    graph.add_tools(HoverTool(renderers=[graph_renderer.node_renderer],tooltips=[("Area", "@label"), ("x","@xs"),("y","@ys")]), TapTool(), BoxSelectTool())


    #Circle(size=15, fill_color=Spectral4[0])
    data_source = graph_renderer.node_renderer.data_source
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

def get_graph():
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