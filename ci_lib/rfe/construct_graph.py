import networkx as nx
import numpy as np

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
        edge_weight = np.zeros((len(selected_feats)))

    if edge_alpha is None:
        edge_alpha = np.ones((len(selected_feats)))



        # matrices to retrieve input/output channels from connections in support network
    mask = np.tri(n_nodes,n_nodes,0, dtype=bool) if feat_type == Feature_Type.UNDIRECTED else np.ones((n_nodes,n_nodes), dtype=bool)
    row_ind = np.repeat(np.arange(n_nodes).reshape([n_nodes,-1]),n_nodes,axis=1)
    col_ind = np.repeat(np.arange(n_nodes).reshape([-1,n_nodes]),n_nodes,axis=0)
    row_ind = row_ind[mask]
    col_ind = col_ind[mask]

    default_color= 'gray'
    selected_color= 'red' #'yellow'

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

        edge_attrs = {}
        for i,ij in enumerate(selected_feats):
            if(col_ind[ij] == row_ind[ij]): #checks for loops
                g.nodes[col_ind[ij]]['selected'] = True
                g.nodes[col_ind[ij]]['color'] = selected_color #colors node red
            else:
                if edge_weight is None:
                    g.add_edge(col_ind[ij],row_ind[ij])
                else:
                    print(color.rgb2hex(plt.cm.viridis(edge_weight[i])))
                    g.add_edge(col_ind[ij],row_ind[ij],edge_color=color.rgb2hex(plt.cm.viridis(edge_weight[i])),edge_alpha=edge_weight[i])
                    #edge_attrs[(col_ind[ij],row_ind[ij])] = plt.cm.viridis(edge_weight[ij])
        print(edge_attrs)
        #nx.set_edge_attributes(g, edge_attrs, "edge_color")
        #Remove nodes with degree 0
        remove = [node for node,degree in dict(g.degree()).items() if degree == 0]
        g.remove_nodes_from(remove)

    return g