import networkx as nx
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors as color

from ci_lib.features import Feature_Type


def construct_rfe_graph(selected_feats, n_nodes, feat_type, edge_weight=None, edge_alpha=None):
    '''
    if labels is None:
        node_labels = dict()
        for i in range(n_nodes): node_labels[i] = i+1
    else:
        node_labels = dict(enumerate(labels))
    '''
    if edge_weight is None:
        edge_weight = 0.5*np.ones((len(selected_feats))) #0.5 := 0 for plt.cm

    if edge_alpha is None:
        edge_alpha = 0.8*np.ones((len(selected_feats)))



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
        #calculate degree
        node_attrs = {node: {"node_size" : 25} for node in g.nodes}
        nx.set_node_attributes(g, node_attrs)

    if feat_type == Feature_Type.DIRECTED or feat_type == Feature_Type.UNDIRECTED:
        g = nx.Graph() if feat_type == Feature_Type.UNDIRECTED else nx.MultiDiGraph()
        for i in range(n_nodes):
            g.add_node(i)
            g.nodes[i]['selected'] = False
            g.nodes[i]['color'] = default_color

        edge_attrs = {}
        for i,ij in enumerate(selected_feats):
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
        node_attrs = {node: {"node_size" : 15+50 * math.sqrt(d/len(selected_feats)) } for node, d in g.degree()}
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