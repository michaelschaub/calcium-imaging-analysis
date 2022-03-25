import pickle as pkl
import numpy as np
from sklearn.metrics import jaccard_score
import pandas

from pathlib import Path
import sys

#define paths
#data_path = / Path('results')

comp=64
feature ="moup_timelags~3_max_components~64"  #"autocovariance_timelags~[3]_max_components~64"
directed=True
n = int(comp*(comp+1)/2) if not directed else int(comp*comp)  #max features cov(64)=2016, moup(64)=4032

rfe_percent=0.2 #0.4
rfe_n = int(np.round((n)*rfe_percent)) #-comp fail

paths_array = ["GN06_2021-01-20_10-15-16","GN06_2021-03-26_10-53-05","GN06_2021-03-29_11-51-27","GN06_2021-01-20_10-15-16_GN06_2021-03-26_10-53-05_GN06_2021-03-29_11-51-27"]
paths = {}

for i in range(len(paths_array)):

    paths["Session "+str(i+1)]= {'feats':Path(__file__).parent.parent.parent.parent  / Path(f"results/{paths_array[i]}/anatomical_ROI~[]/All/Decoding/rfe/left_visual.right_visual.left_tactile.right_tactile.left_vistact.right_vistact/{rfe_n}/{feature}/best_feats.pkl"),
                                 'perf':Path(__file__).parent.parent.parent.parent  / Path(f"results/{paths_array[i]}/anatomical_ROI~[]/All/Decoding/rfe/left_visual.right_visual.left_tactile.right_tactile.left_vistact.right_vistact/{rfe_n}/{feature}/decoder_perf.pkl")}



print("paths",paths)

#Load best features & perf
best_feats = {}
perfs = {}

for name,path in paths.items():
    with open(path['feats'], 'rb') as f:
        feat_inds = pkl.load(f) #np.random.randint(64,size=13)
        print(len(feat_inds))
        print(n)
        #transform features to binary vectors
        binary_feats = np.zeros((n))
        binary_feats[feat_inds] = 1
        best_feats[name] = binary_feats

    with open(path['perf'], 'rb') as f:
        perf =  pkl.load(f)
        print(perf)
        perfs[name] = perf
        print(perfs)


    rng = np.random.default_rng()

if rfe_n=="full": rfe_n=n
random_index = rng.choice(n, size=rfe_n, replace=False)
print(random_index)
binary_feats = np.zeros((n))
binary_feats[random_index] = 1
best_feats['random'] = binary_feats



print(best_feats)

#compute pairwise jaccard score
j_scores = np.full((len(best_feats),len(best_feats)+1), np.nan)

for x, (name_x , best_feat_x) in enumerate(best_feats.items()):
    for y, (name_y , best_feat_y) in enumerate(best_feats.items()):
        if(x<=y):
            j_scores[x,y] = jaccard_score(best_feat_x,best_feat_y)



print(j_scores)

print(perfs)

j_scores[:len(paths),-1]= np.array(list(perfs.values())).astype(float)[:,0]
print(j_scores)
cols = list(best_feats.keys())
cols.append("Accuracy")

print(cols)
df = pandas.DataFrame(data=j_scores,index=best_feats.keys(), columns=cols)

print(df)

table = df.to_latex(na_rep='',float_format='{:,.2%}'.format,caption=f"Jaccard Index for the {int(rfe_percent*100)}\% most decisive connections ({rfe_n} /{n}  )")

#touch()
#with open('mytable.tex', 'w') as tf:
#    tf.write(table)
print(table)