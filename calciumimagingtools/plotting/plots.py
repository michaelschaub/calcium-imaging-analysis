import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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


def graph_circle_plot(list_best_feat, n_nodes, title,type_measure=1,save_path=False, directed=False):
    #%% network and plot properties
    N = n_nodes #20 # number of nodes
    print(N)
    # positions for circular layout with origin at bottoms
    pos_circ = dict()
    for i in range(N):
        pos_circ[i] = np.array([np.sin(2*np.pi*(i/N+0.5/N)), np.cos(2*np.pi*(i/N+0.5/N))])

    # channel labels
    ch_labels = dict()
    for i in range(N):
        ch_labels[i] = i+1

    # matrices to retrieve input/output channels from connections in support network
    mask = np.tri(N,N,0, dtype=bool) if not directed else np.ones((N,N), dtype=bool)

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
    if type_measure == 0: # nodal
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
        nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
    else: # interactional
        #list_best_feat = np.argsort(class_perfs.mean(0))[:n_edges] # select n best features
        g = nx.Graph() if not directed else nx.MultiDiGraph()
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
        nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
        nx.draw_networkx_edges(g,pos=pos_circ,edgelist=g.edges(),edge_color='#009DAE')
    if (not save_path):
        plt.show()
    else:
        plt.savefig(save_path)