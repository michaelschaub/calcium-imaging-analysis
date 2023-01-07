import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import seaborn as sns
import pandas as pd

from pathlib import Path
import sys

sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log


### Setup
logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    decoders = snakemake.params['decoders']
    conditions = snakemake.params['conds']
    features = snakemake.params['features']
    feature_labels = [feat.split('_')[0] for feat in features]
    #    try:
    #        param = feat.split('[', 1)[1].split(']')[0] # re.search(r"\[([A-Za-z0-9_',]+)\]", feat)
    #        features.append(feature) if param == "" else features.append("_".join([feature,param]))
    #    except:
    #        features.append(feature)
    parcellations = snakemake.params['parcellations']

    #Load flattened input array & reconstruct dims
    perfs = np.empty((len(list(snakemake.input))),dtype=object) #perfs of diff. features can have diff. dim
    for i,file in enumerate(snakemake.input):
        with open(file, "rb") as f:
            perfs[i] = np.array(pickle.load(f)).flatten() #Flatten timepoints x reps to eval_point (last variable dim as object in perf)
    perfs = perfs.reshape((len(features),len(parcellations),len(decoders)))

    #Construct long df
    perfs_dict = [{"decoder": decoder, "eval_point": i, "accuracy": perf, "parcellation":parcel, "feature":feat} 
                    for d,decoder in enumerate(decoders)
                    for p, parcel in enumerate(parcellations)
                    for f, feat in enumerate(feature_labels)
                    for i,perf in enumerate(perfs[f,p,d])]
    perfs_df =  pd.json_normalize(perfs_dict)

    #Define hue for parcellations & decoder
    hue=perfs_df[['parcellation', 'decoder']].apply(lambda row: f"{row.parcellation}{' (Shuffle)' if 'shuffle' in row.decoder else ''}", axis=1)
    hue.name = 'Parcellation'

    #decoders = [ dec.split('_')[0] for dec in snakemake.params['decoders']]
    sns.set(rc={'figure.figsize':(15,7.5)})
    sns.set_style("whitegrid")

    #group of features
    activity_df = perfs_df[perfs_df["feature"].str.contains("activity")]
    FC_df = perfs_df.loc[perfs_df["feature"].str.contains("FC")]
    EC_df = perfs_df.loc[perfs_df["feature"].str.contains("moup")]

    # = pd.merge(pd.merge(activity_df,FC_df,how="outer",on=["feature"]), EC_df ,how="outer",on=["feature"])
    #rest_df = #perfs_df.iloc[perfs_df.index.difference(union_df.index)]

    group_feats = [activity_df,FC_df,EC_df] 
    rest_df = pd.concat([perfs_df]+group_feats).drop_duplicates(keep=False)
    group_feats.append(rest_df)
    group_labels = ["Activity","Functional Connectivity","Effective Connectivity","Other"]

    for i,df in enumerate(group_feats):
        if df.empty:
            del group_feats[i]
            del group_labels[i]

    group_n = [len(df['feature'].unique()) for df in group_feats]
    
    #[len([feat for feat in features if "activity" in feat]),len([feat for feat in features if "FC" in feat]),len([feat for feat in features if "moup" in feat])] #better

    print(group_n)

    fig, axs = plt.subplots(1, len(group_feats), sharey=True, gridspec_kw={'width_ratios':group_n}) #, width_ratios=np.array(group_n,dtype=int))
    j=0

    for i,ax in enumerate(axs):
        sns.violinplot(data=group_feats[i], x="feature", y="accuracy", hue=hue, cut=0, palette='deep',linewidth=1,saturation=1,ax=axs[i],legend=None)

        #Axis labels
        axs[i].set_xlabel(group_labels[i])
        axs[i].set_ylabel("Accuracy" if i==0 else "")

        #Legends
        if i<len(group_feats)-1:
            axs[i].get_legend().remove()


        #Remove left spine (True), except for first element (False)
        #Remove right spine(True), except for last element (False) / or always
        sns.despine(ax=axs[i],left=(i>0),right=True) #(i<len(group_feats)-1))



    #for violin in axs.collections[::2]:
    #    violin.set_alpha(0.9)

        #Draw random chance
        axs[i].plot([-.5, group_n[i]-.5], [1/len(conditions), 1/len(conditions)], '--k')
        axs[i].text(group_n[i]-0.8,1/len(conditions)+0.01,'random chance',fontsize=7,ha='center')

        #Draw background
        for f in range(group_n[i]):
            if (j % 2) == 1:
                axs[i].axvspan(f-0.5, f+0.5, facecolor='0.2', alpha=0.05, zorder=-100)
            j += 1

    plt.subplots_adjust(wspace=0, hspace=0)

    '''
    results = [] #"\n".join(feat.split('_')[0:2]) if len(feat.split('_'))>3 else feat.split('_')[0] for feat in snakemake.params["features"]]

    for feat in snakemake.params["features"]:
        feature = feat.split('_')[0]
        try:
            param = feat.split('[', 1)[1].split(']')[0] # re.search(r"\[([A-Za-z0-9_',]+)\]", feat)
            features.append(feature) if param == "" else features.append("_".join([feature,param]))
        except:
            features.append(feature)

    #features = [ "\n".join(feat.split('_')) for feat in snakemake.params["features"]]

    dec_n = len(decoders)

    plt.figure(figsize=[2.4+len(features),4.8],dpi=600)


    #Move to util
    def bold(text):
        return r"$\bf{" + text + "}$"
    #Move to util
    linebreak= '\n'
    subject_str = bold("Subjects(")+'#'+bold("Sessions)")+f": {', '.join(snakemake.params['subjects'])}"

    title_str= ""
    x_str=""

    if 'parcellation' in snakemake.wildcards.keys():
        title_str= bold("Parcellation")+f": {snakemake.wildcards['parcellation']}"
        x_str = "Features"
    elif "feature" in snakemake.wildcards.keys():
        title_str= bold("Feature")+f": {snakemake.wildcards['feature']}"
        x_str = "Parcellations"

    conditions_str = bold("Conditions")+f":{linebreak}{linebreak.join(snakemake.params['conds'])}"
    trials_str = bold("Trials")+f":{linebreak}{linebreak.join([f'{k}: {v}'  for k,v in snakemake.params['trials'].items()])}"




    colors = cm.get_cmap('Accent',len(decoders))

    violin_plts= np.empty([len(features),len(decoders)],dtype='object')
    for f, feature in enumerate(features):
        if (f % 2) == 1:
            plt.axvspan(f-0.5, f+0.5, facecolor='0.2', alpha=0.05, zorder=-100)

        for d,decoder in enumerate(decoders):

            with open(snakemake.input[d+f*dec_n], "rb") as file:
                try:
                    perf = pickle.load(file)
                    #Has dimension timepoints x reps
                    perf = np.array(perf).flatten()
                except:
                    perf = [0]
                violin_plts[f,d]=plots.colored_violinplot(perf, positions=f + np.arange(1) + ((d+1)*1/(dec_n+1))-0.5, widths=[1/(dec_n+2)], color=colors(d/dec_n))
    
    #plt.legend( [colors(i/len(decoders)) for i in range(len(decoders))], decoders )
    plt.legend( [ v['bodies'][0] for v in violin_plts[0]], decoders,loc='lower left')

    
    

    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(range(len(features)), features,fontsize=10)
    plt.xlabel(x_str, fontsize=14)
    '''
    #ax = plt.gca()
    #ax.set_ylim([0, 1])
    #plt.subplots_adjust(left=0.2)
    plt.savefig( snakemake.output[0] )


    #plt.gcf().text(0.78, 0.5, "\n \n".join([conditions_str,trials_str]) , fontsize=10, va='center')
    #plt.subplots_adjust(right=0.75)
    #plt.suptitle("\n".join([title_str,subject_str]),x=0.45)


    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
