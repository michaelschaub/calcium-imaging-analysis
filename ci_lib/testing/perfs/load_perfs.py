import pickle as pkl
import numpy as np
from sklearn.metrics import jaccard_score
import pandas

from pathlib import Path
import sys

#parcellation="anatomical_ROI~['VIS,SS']"
parcellation="SVD" #anatomical_ROI~[]"
comp= 64 #36

#feature ="moup_timelags~1_max_components~64"
#feature ="autocovariance_timelags~[1]_max_components~64"
feature = "mean" #_max_components~64"

decoder = "LDA_reps~50"

#directed=True
directed = False
n = int(comp*(comp+1)/2) if not directed else int(comp*comp)  #max features cov(64)=2016, moup(64)=4032
if "mean" in feature:
    n = comp



#paths_array = ["GN06_2021-01-20_10-15-16","GN06_2021-03-26_10-53-05","GN06_2021-03-29_11-51-27"] #,"GN06_2021-03-29_11-51-27_GN06_2021-03-26_10-53-05"]

paths_array = ["GN10_2021-03-29_15-41-22","GN10_2021-04-14_14-50-49","GN10_2021-04-21_11-00-51"]

paths = {}

for i in range(len(paths_array)):

    paths["Session "+str(i+1)]= {'perf':Path(__file__).parent.parent.parent.parent  / Path(f"results/{paths_array[i]}/{parcellation}/All/Decoding/decoder/left_visual.right_visual.left_tactile.right_tactile.left_vistact.right_vistact/{feature}/{decoder}/decoder_perf.pkl")}



print("paths",paths)

#Load best features & perf
best_feats = {}
perfs = {}
q = np.zeros((4,len(paths)))

for i,(name,path) in enumerate(paths.items()):


    with open(path['perf'], 'rb') as f:
        perf =  pkl.load(f)
        print(perf)

        q[0,i]=np.quantile(perf,0.25)
        q[1,i]=np.quantile(perf,0.5)
        q[2,i]=np.quantile(perf,0.75)
        q[3,i]=np.mean(perf)

        perfs[name] = {
            'q1': np.quantile(perf,0.25),
            'median': np.quantile(perf,0.5),
            'q3': np.quantile(perf,0.75)
        }
        print(perfs)




print(q)
print(cols)
df = pandas.DataFrame(data=j_scores,index=best_feats.keys(), columns=cols)

print(df)

table = df.to_latex(na_rep='',float_format='{:,.2%}'.format,caption=f"Jaccard Index for the {int(rfe_percent*100)}\% most decisive connections ({rfe_n}/{n})")

#touch()
#with open('mytable.tex', 'w') as tf:
#    tf.write(table)
print(table)