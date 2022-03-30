import pickle as pkl
import numpy as np
from sklearn.metrics import jaccard_score
import pandas

from pathlib import Path
import sys

#parcellation="anatomical_ROI~['VIS,SS']"
parcellation="anatomical_ROI~[]"
comp= 64 #36

#feature ="moup_timelags~1_max_components~64"
#feature ="autocovariance_timelags~[1]_max_components~64"
feature = "mean" #_max_components~64"
mirror = False #True #False #

#directed=True
directed=False
n = int(comp*(comp+1)/2) if not directed else int(comp*comp)  #max features cov(64)=2016, moup(64)=4032
if "mean" in feature:
    n = comp


rfe_percent=0.1 #0.4
rfe_n = int(np.round((n)*rfe_percent)) #-comp fail

paths_array = ["GN06_2021-01-20_10-15-16","GN06_2021-03-26_10-53-05","GN06_2021-03-29_11-51-27"]#,"GN06_2021-03-29_11-51-27_GN06_2021-03-26_10-53-05"]

paths_array = ["GN10_2021-03-29_15-41-22","GN10_2021-04-21_11-00-51","GN10_2021-04-14_14-50-49"]

paths = {}

for i in range(len(paths_array)):

    paths["Session "+str(i+1)]= {'feats':Path(__file__).parent.parent.parent.parent  / Path(f"results/{paths_array[i]}/{parcellation}/All/Decoding/rfe/left_visual.right_visual.left_tactile.right_tactile.left_vistact.right_vistact/{rfe_percent}/{feature}/best_feats.pkl"),
                                 'perf':Path(__file__).parent.parent.parent.parent  / Path(f"results/{paths_array[i]}/{parcellation}/All/Decoding/rfe/left_visual.right_visual.left_tactile.right_tactile.left_vistact.right_vistact/{rfe_percent}/{feature}/decoder_perf.pkl")}



print("paths",paths)

#Load best features & perf
best_feats = {}
perfs = {}

for name,path in paths.items():
    with open(path['feats'], 'rb') as f:
        feat_inds = pkl.load(f) #np.random.randint(64,size=13)
        print((feat_inds))
        print(n)
        #transform features to binary vectors
        binary_feats = np.zeros((n),dtype=np.bool_)
        binary_feats[feat_inds] = 1
        best_feats[name] = binary_feats

        #mirrors means for both sides
        if mirror:
            h = int(n/2)
            print(binary_feats)
            binary_feats = np.bitwise_or(binary_feats,np.flip(binary_feats))
            #binary_feats[-h:] = np.bitwise_or(np.flip(binary_feats[:h]),binary_feats[:-n])
            print(binary_feats)

        best_feats[name] = binary_feats

    with open(path['perf'], 'rb') as f:
        perf =  pkl.load(f)
        print(perf)
        perf = np.mean(perf)
        print(perf)
        perfs[name] = perf
        print(perfs)



rng = np.random.default_rng()
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

j_scores[:len(paths),-1]= np.array(list(perfs.values())).astype(float)
print(j_scores)
cols = list(best_feats.keys())
cols.append("Accuracy")

print(cols)
df = pandas.DataFrame(data=j_scores,index=best_feats.keys(), columns=cols)

print(df)

table = df.to_latex(na_rep='',float_format='{:,.2%}'.format,caption=f"Jaccard Index for the {int(rfe_percent*100)}\% most decisive connections ({rfe_n}/{n})")

#touch()
#with open('mytable.tex', 'w') as tf:
#    tf.write(table)
print(table)