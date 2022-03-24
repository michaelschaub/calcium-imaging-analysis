import pickle as pkl
import numpy as np
from sklearn.metrics import jaccard_score
import pandas

#define paths
n = 64 #max features cov(64)=2016, moup(64)=4032

feature = "autocovariance_timelags~[1]_max_components~64"
rfe_n = "403"
sessions_n = 4

paths = {}
for i in range(1,sessions_n+1):
    paths["Session "+str(i)]= "data/session"+str(i)+"/"+rfe_n+"/"+feature+"/best_feats.pkl"

#    "Session 1":
#    "Session 2": "",
#    "Session 3": "",
#    "Session 4": "",
#}

#Load best features
best_feats = {}
for name,path in paths.items():
    #with open(path, 'rb') as f:
        feat_inds = np.random.randint(64,size=13) #pickle.load(f)
        #transform features to binary vectors
        binary_feats = np.zeros((n))
        binary_feats[feat_inds] = 1
        best_feats[name] = binary_feats

best_feats['random'] = np.random.randint(2,size=64)

print(best_feats)

#compute pairwise jaccard score
j_scores = np.zeros((len(best_feats),len(best_feats)))

for x, (name_x , best_feat_x) in enumerate(best_feats.items()):
    for y, (name_y , best_feat_y) in enumerate(best_feats.items()):
        if(x<=y):
            j_scores[x,y] = jaccard_score(best_feat_x,best_feat_y)
        else:
            j_scores[x,y] = np.nan

print(j_scores)

df = pandas.DataFrame(data=j_scores,index=best_feats.keys(), columns=best_feats.keys())

print(df)

table = df.to_latex(na_rep='',float_format='{:,.2%}'.format,caption="Jaccard Index for the 20\% most decisive connections ( /  )")

print(table)