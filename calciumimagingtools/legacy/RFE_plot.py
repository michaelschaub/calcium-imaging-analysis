#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:34:37 2018


"""

import numpy as np
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
import networkx as nx



#%% network and plot properties

N = 20 # number of nodes
    
# positions for circular layout with origin at bottoms
pos_circ = dict()
for i in range(N):
    pos_circ[i] = np.array([np.sin(2*np.pi*(i/N+0.5/N)), np.cos(2*np.pi*(i/N+0.5/N))])
    
# channel labels
ch_labels = dict()
for i in range(N):
    ch_labels[i] = i+1

# matrices to retrieve input/output channels from connections in support network
mask_tri = np.tri(N,N,-1, dtype=bool)
row_ind = np.repeat(np.arange(N).reshape([N,-1]),N,axis=1)
col_ind = np.repeat(np.arange(N).reshape([-1,N]),N,axis=0)    
row_ind = row_ind[mask_tri]
col_ind = col_ind[mask_tri]


#%% classifier and learning parameters

# MLR adapted for recursive feature elimination (RFE)
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])
 
# cross-validation scheme
cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
n_rep = 10 # number of repetitions

# RFE wrappers
RFE_node = skfs.RFE(c_MLR,n_features_to_select=1)
RFE_inter = skfs.RFE(c_MLR,n_features_to_select=int(N/2))

# record classification performance 
rk_node = np.zeros([n_rep,N],dtype=np.int) # RFE rankings for node-type measures (N feature)
rk_inter = np.zeros([n_rep,int(N*(N-1)/2)],dtype=np.int) # RFE rankings for interaction-type measures (N(N-1)/2 feature)


#%% recursive fetaure elimination (RFE)

# generate random data and labels for 2 categories
S = 100 # number of samples (e.g. trials, etc.)

type_measure = 1 # 0: node; 1: interactions like FC

if type_measure == 0:
    vect_features = np.random.rand(S,N)
    vect_features[:int(S/2),:] += np.outer(np.ones(int(S/2)), np.arange(N)/N) # bias that differ across features for 1st class (half of samples)
else:
    vect_features = np.zeros([S,N,N])
    for s in range(S):
        W = np.eye(N)
        if s<S/2:
            W[:int(N/2),:int(N/2)] += np.random.rand(int(N/2),int(N/2))
        ts_tmp = np.dot(W, np.random.rand(N, 500)) # random time series with and without correlations (for 1st half of nodes)
        vect_features[s,:,:] = np.corrcoef(ts_tmp, rowvar=True)
    vect_features = vect_features[:,mask_tri] # retain only a triangle from whole matrix

print(vect_features.shape)
# labels: 2 classes with S/2 samples each
labels = np.zeros([S], dtype=int)
labels[int(S/2):] = 1
print(labels.shape)
# loop over repetitions (train/test sets)
for i_rep in range(n_rep):
    
    for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1 
        
        # RFE for MLR
        if type_measure == 0: # node-wise feature
            RFE_node.fit(vect_features[ind_train,:], labels[ind_train])
            rk_node[i_rep,:] = RFE_node.ranking_
        else: # interaction-wise feature
            RFE_inter.fit(vect_features[ind_train,:], labels[ind_train])
            print(RFE_inter.ranking_.shape)
            rk_inter[i_rep,:] = RFE_inter.ranking_



#%% plots
fmt_grph = 'png'



# plot RFE support network
plt.figure(figsize=[10,10])
plt.axes([0.05,0.05,0.95,0.95])
plt.axis('off')
if type_measure == 0: # nodal
    list_best_feat = np.argsort(rk_node.mean(0))[:10] # select 10 best features
    node_color_aff = []
    g = nx.Graph()
    for i in range(N):
        g.add_node(i)
        if i in list_best_feat:
            node_color_aff += ['red']
        else:
            node_color_aff += ['orange']
    nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
    nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
else: # interactional
    list_best_feat = np.argsort(rk_inter.mean(0))[:20] # select 20 best features
    g = nx.Graph()
    for i in range(N):
        g.add_node(i)
    node_color_aff = ['orange']*N
    list_ROI_from_to = [] # list of input/output ROIs involved in connections of support network
    for ij in list_best_feat:
        g.add_edge(col_ind[ij],row_ind[ij])
    print(g)
    nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
    nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
    nx.draw_networkx_edges(g,pos=pos_circ,edgelist=g.edges(),edge_color='r')

plt.show()

        
        
