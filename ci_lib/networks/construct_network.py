import networkx as nx
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors as color

from ci_lib.features import Feature_Type

def construct_network_from_feat(feat_ind, n_nodes, feat_type, feat_val = None, node_label=None):
    """
    :param feature:
    :param n_nodes:
    :type n_nodes: int
    :param feat_type: Type of Feature
    :type feat_type: {Feature_Type}
    """

    #Obtain adj matrix + nodes

    feat_ind = np.asarray(feat_ind)

    nodes = np.zeros((n_nodes))
    adj = np.zeros((n_nodes,n_nodes))

    if feat_val is not None:
        node_val = np.zeros((n_nodes))
        edge_val = np.zeros((n_nodes,n_nodes))
    else:
        node_val = None
        edge_val = None

    if feat_ind.ndim == 1:
        if feat_type == Feature_Type.NODE:

            nodes[feat_ind] = 1
            if feat_val is not None:
                node_val[feat_ind] = feat_val

            #Add self-loop
            adj[np.eye(n_nodes)] = nodes


        if feat_type == Feature_Type.UNDIRECTED or feat_type == Feature_Type.DIRECTED:
            mask = np.tri(n_nodes,n_nodes,0, dtype=bool) if feat_type == Feature_Type.UNDIRECTED else np.ones((n_nodes,n_nodes), dtype=bool)
            row_ind = np.repeat(np.arange(n_nodes).reshape([n_nodes,-1]),n_nodes,axis=1)
            col_ind = np.repeat(np.arange(n_nodes).reshape([-1,n_nodes]),n_nodes,axis=0)
            row_ind = row_ind[mask]
            col_ind = col_ind[mask]

            # ij = index of selected features in flattened adj matrix, i index of that index
            for i,ij in enumerate(feat_ind):
                #Mark nodes & edges as connected/selected in the node & adj matrix
                nodes[col_ind[ij]] = 1
                nodes[row_ind[ij]] = 1
                adj[col_ind[ij],row_ind[ij]] = 1

                if feat_val is not None:
                    #Attach feat value to edge
                    edge_val[col_ind[ij],row_ind[ij]] = feat_val[i]

                    #Attach values for self-loops to node
                    if col_ind[ij] == row_ind[ij]:
                        node_val[col_ind[ij]] = feat_val[i]

                #Mirror for undirected (check if needed)
                if feat_type == Feature_Type.UNDIRECTED:
                    adj[row_ind[ij],col_ind[ij]] = 1
                    if feat_val is not None:
                        edge_val[row_ind[ij],col_ind[ij]] = feat_val[i]

    return construct_graph(nodes, adj, node_val, edge_val, node_label, feat_type)



def construct_graph(nodes, adj, node_val=None, edge_val=None, node_label=None, feat_type=None):
    dt = [("value",float)]

    edges = edge_val if edge_val is not None else adj

    if not (edges==edges.T).all() or feat_type == Feature_Type.DIRECTED:
        #Is not symmetric, assumed to be undirected or epxlicitly defined as directed
        g = nx.from_numpy_matrix(edges, create_using=nx.MultiDiGraph())
    else:
        g = nx.from_numpy_matrix(edges)

    if node_val is not None:
        node_val = {n: {"weight" : node_val[n]} for n in g.nodes}
        nx.set_node_attributes(g,node_val)
    if node_label is not None:
        node_label = {n: {"label" : node_label[n]} for n in g.nodes}
        nx.set_node_attributes(g,node_label)

    return g

def weight_to_color(weight):
    print("weight")
    print(weight)
    c = color.rgb2hex(plt.cm.viridis(weight))
    print(c)
    return c

def weight_to_alpha(weight):
    if weight < 0.5:
        return 0
    return weight

def add_bokeh_attributes(g,default_color='grey',highlight_color='yellow'):

    #Self loops can't be displayed, embedd it as node property
    adj = nx.to_numpy_array(g)
    print(adj)
    dia = np.diagonal(adj) #[np.eye((nodes),dtype=bool)]
    print(dia)
    self_loop = dia > 0
    print(len(self_loop))
    print(len(g.nodes))
    self_loop = {n: {"selected" : self_loop[n]} for n in g.nodes}
    nx.set_node_attributes(g,self_loop)

    #node size
    node_attrs = {node: {"node_size" : 25} for node in g.nodes}
    nx.set_node_attributes(g, node_attrs)

    #

    #set node color
    for n in g.nodes():
        if 'weight' in g.nodes[n]:
            g.nodes[n]['color'] = weight_to_color(g.nodes[n]['weight'])
        else:
            g.nodes[n]['color'] = highlight_color if g.nodes[n]["selected"] else default_color

    #set edge color
    for edge in g.edges():
        src, trg = edge

        if isinstance(g,nx.MultiDiGraph):
            print(g[src][trg][0])

            if 'weight' in g[src][trg][0]:
                g[src][trg][0]["edge_color"] = weight_to_color(g[src][trg][0]['weight'])
                g[src][trg][0]["edge_alpha"] = weight_to_alpha(g[src][trg][0]['weight'])
            else:
                g[src][trg][0]["edge_color"] = highlight_color
                g[src][trg][0]["edge_alpha"]= 1
        else:
            if 'weight' in g[src][trg]:
                g[src][trg]["edge_color"] = weight_to_color(g[src][trg]['weight'])
                g[src][trg]["edge_alpha"] = weight_to_alpha(g[src][trg]['weight'])
            else:
                g[src][trg]["edge_color"] = highlight_color
                g[src][trg]["edge_alpha"]= 1


    return g


def construct_network(edges, n_nodes, feat_type, edge_weight=None, edge_alpha=None):


    #if labels is None:
    #    node_labels = dict()
    #    for i in range(n_nodes): node_labels[i] = i+1
    #else:
    #    node_labels = dict(enumerate(labels))
    if edge_weight is None:
        edge_weight = 0.5*np.ones((len(edges))) #0.5 := 0 for plt.cm

    if edge_alpha is None:
        edge_alpha = 0.8*np.ones((len(edges)))

    #if isinstance(edges, np.ndarray):
    #    if edges.ndim == 2:
    #        adj_matrix = edges



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
            g.nodes[i]['selected'] = (i in edges)
            g.nodes[i]['color'] = selected_color if (i in edges) else default_color
        #g.nodes[selected_feats]['color'] = selected_color if (i in selected_feats) else default_color
        #calculate degree
        node_attrs = {node: {"node_size" : 25} for node in g.nodes}
        nx.set_node_attributes(g, node_attrs)

    if feat_type == Feature_Type.DIRECTED or feat_type == Feature_Type.UNDIRECTED:
        g = nx.Graph() if feat_type == Feature_Type.UNDIRECTED else nx.MultiDiGraph()
        print((feat_type))
        print(feat_type == Feature_Type.UNDIRECTED)

        for i in range(n_nodes):
            g.add_node(i)
            g.nodes[i]['selected'] = False
            g.nodes[i]['color'] = default_color

        edge_attrs = {}
        for i,ij in enumerate(edges):
            if(col_ind[ij] == row_ind[ij]): #checks for loops
                g.nodes[col_ind[ij]]['selected'] = True
                g.nodes[col_ind[ij]]['color'] = selected_color #colors node red
            else:
                if edge_weight is None:
                    g.add_edge(col_ind[ij],row_ind[ij])
                else:
                    g.add_edge(col_ind[ij],row_ind[ij],edge_color=color.rgb2hex(plt.cm.viridis(edge_weight[i])),edge_alpha=edge_alpha[i])
                    #edge_attrs[(col_ind[ij],row_ind[ij])] = plt.cm.viridis(edge_weight[ij])
        #nx.set_edge_attributes(g, edge_attrs, "edge_color")

        #calculate degree
        node_attrs = {node: {"node_size" : 15+50 * math.sqrt(d/len(edges)) } for node, d in g.degree()}
        nx.set_node_attributes(g, node_attrs)

        #Remove nodes with degree 0
        remove = [node for node,degree in dict(g.degree()).items() if degree == 0]
        g.remove_nodes_from(remove)

        if feat_type == Feature_Type.DIRECTED:
            strong_comps = nx.weakly_connected_components(g)
        else:
            strong_comps = nx.connected_components(g)

        generator_list = list(strong_comps) #otherwise loop is empty / generators can only be tierated once..
        #n_comps= sum(1 for _ in generator_copy) #cause generators don't have a length...
        print("comps",len(generator_list))
        for i,c in enumerate(generator_list):
            print(c)
            for node in c:
                g.nodes[node]['color'] = color.rgb2hex(plt.cm.tab20(i/len(generator_list)))


            for edge in g.edges(c):

                src, trg = edge

                if feat_type == Feature_Type.DIRECTED:
                    g[src][trg][0]["edge_color"] = color.rgb2hex(plt.cm.tab20(i/len(generator_list)))
                else:
                    g[src][trg]["edge_color"] = color.rgb2hex(plt.cm.tab20(i/len(generator_list)))


    return g