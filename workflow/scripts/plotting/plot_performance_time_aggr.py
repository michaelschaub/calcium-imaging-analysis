import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import yaml

import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

import ci_lib.plotting as plots
from ci_lib.utils import snakemake_tools
from ci_lib.utils.logging import start_log

logger = start_log(snakemake)
try:
    timer_start = snakemake_tools.start_timer()

    #Loading
    for 
    with open(snakemake.input["perf_df"], "rb") as f:
        df = pickle.load(f)

            
    decoders = [ dec.split('_')[0] for dec in snakemake.params['decoders']]
    conditions = snakemake.params['conds']
    #parcellation = None

    

    #Plotting
    
    font_scale=1.1
    sns.set(font_scale=font_scale)
    sns.set(rc={'figure.figsize':(15,6)})
    sns.set_style("whitegrid",{'axes.xaxis.grid' : False})



    #Plotting across parcellations

    #

    #Plotting across features (potentially diff dims!)

    #Plott across datasets



    if 'parcellations' in snakemake.params.keys():
        perf = np.asarray(perf,dtype=float)
        _, timepoints_n, runs_n = perf.shape

        parcellation = snakemake.params['parcellations'] 
        perf = perf.reshape(len(decoders),len(parcellation), timepoints_n, runs_n)
    
        #Create list of dicts 
        accuracy_dict = np.asarray([[[[{"decoder": decoder, "t": t, "run":r, "accuracy":run_perf, "Parcellation":parcel} for r, run_perf in  enumerate(timepoint_perf)] for t,timepoint_perf in enumerate(perf[i,p,:,:])] for i,decoder in enumerate(decoders)] for p, parcel in enumerate(parcellation)]).flatten()
        accuracy_df =  pd.json_normalize(accuracy_dict)
        

        #treat shuffle like additional parcellation
        shuffle_df = accuracy_df[accuracy_df['decoder'].str.contains('shuffle')]
        non_shuffle_df = pd.concat([accuracy_df, shuffle_df]).drop_duplicates(keep=False)

        shuffle_df['Parcellation'] = "Shuffle (∀Parcellations)" 
        accuracy_df = pd.concat([non_shuffle_df,shuffle_df])

        #hue=accuracy_df[['parcellation', 'decoder']].apply(lambda row: f"{row.parcellation}{' (Shuffle)' if 'shuffle' in row.decoder else ''}", axis=1)
        #hue.name = 'Parcellation'

        #hue_order=hue_order

        accuracy_over_time_plot  = sns.lineplot(data=accuracy_df, x="t", y="accuracy",hue="Parcellation",errorbar=('pi',90),palette='deep', err_kws={"alpha":0.25})

    #Plotting across features (potentially diff dims!)
    elif "features" in snakemake.params.keys():
        
        features = snakemake.params['features'] 
        #perf is flatt with feature x decoder
        perf = [[perf[int(f+d*len(features))] for f in range(len(features))] for d in range(len(decoders))] #can't reshape cause last dim is different


        for j,decoder_perf in enumerate(perf):
            for i,feat in enumerate(features):

                if "phase" in snakemake.config["features"][feat].keys():
                    runs_n = len(perf[j][i][0])
                    total_timesteps = int(snakemake.config["phase"]["all"]["stop"]) - int(snakemake.config["phase"]["all"]["start"])  
                    start = int(snakemake.config["phase"][snakemake.config["features"][feat]["phase"]]["start"])
                    stop = int(snakemake.config["phase"][snakemake.config["features"][feat]["phase"]]["stop"]) 
                    tmp = np.zeros((total_timesteps,runs_n))
                    tmp[:] = np.inf #to avoid being plotted by seaborn
                    #print(stop)
                    #print(total_timesteps)
                    
                    tmp[start:stop,:]= decoder_perf[i] if len(decoder_perf)>1 else decoder_perf[i][0] #[:] #np.asarray(np.repeat([decoder_perf[i][0]],stop-start,axis=0))

                    #print(tmp)
                    perf[j][i] = tmp
                    print(feat)
                    print(len(perf[j][i]))
                    print(perf[j][i][0])
        #print(perf)

        perf = np.asarray(perf,dtype=float) #TODO this will break if feat have different timesteps & not defined by phases


        _ , _, timepoints_n, runs_n = perf.shape
        #perf = perf.reshape(len(decoders),len(features), timepoints_n, runs_n)
    
        #Create list of dicts 
        accuracy_dict = np.asarray([[[[{"decoder": decoder, "t": t, "run":r, "accuracy":run_perf, "Feature":feature} for r, run_perf in  enumerate(timepoint_perf)] for t,timepoint_perf in enumerate(perf[i,f,:,:])] for i,decoder in enumerate(decoders)] for f, feature in enumerate(features)]).flatten()
        accuracy_df =  pd.json_normalize(accuracy_dict)

        #remove infs from padded features
        accuracy_df = accuracy_df[accuracy_df.accuracy != np.inf]

        #treat shuffle like additional parcellation
        shuffle_df = accuracy_df[accuracy_df['decoder'].str.contains('shuffle')]
        non_shuffle_df = pd.concat([accuracy_df, shuffle_df]).drop_duplicates(keep=False)

        shuffle_df['Feature'] = "Shuffle (∀Features)" 
        accuracy_df = pd.concat([non_shuffle_df,shuffle_df])

        print(accuracy_df)
        #not needed when shuffle combined
        #hue=accuracy_df[['feature', 'decoder']].apply(lambda row: f"{row.feature}{' (Shuffle)' if 'shuffle' in row.decoder else ''}", axis=1)
        #hue.name = 'Features'
        accuracy_over_time_plot  = sns.lineplot(data=accuracy_df, x="t", y="accuracy",hue="Feature",errorbar=('pi',90)) #hue,errorbar=('pi',90))

    elif "clusters" in snakemake.params.keys():
        perf = np.asarray(perf,dtype=float)
        logger.info(perf.shape)
        timepoints_n, n_models, runs_n = perf[0].shape

        accuracy_dict = np.asarray([{"t": t, "cluster":c, "run":rep, "accuracy":accuracy} for t,perf_t in enumerate(perf[0]) for c,cluster in enumerate(perf_t) for rep,accuracy in enumerate(cluster)])
        logger.info(accuracy_dict)
        accuracy_df =  pd.json_normalize(accuracy_dict)
        

        accuracy_over_time_plot  = sns.lineplot(data=accuracy_df, x="t", y="accuracy",hue="cluster",errorbar=('pi',90),legend="full",palette= "tab20") 

    #Plotting for 1 parcellation and 1 feature
    else:
        perf = np.asarray(perf,dtype=float)
        _, timepoints_n, runs_n = perf.shape

        #Create list of dicts
        accuracy_dict = np.asarray([[[{"decoder": decoder, "t": t, "run":r, "accuracy":run_perf} for r, run_perf in  enumerate(timepoint_perf)] for t,timepoint_perf in enumerate(perf[i,:,:])] for i,decoder in enumerate(decoders)]).flatten()
        accuracy_df =  pd.json_normalize(accuracy_dict)

        accuracy_over_time_plot  = sns.lineplot(data=accuracy_df, x="t", y="accuracy",hue="decoder",errorbar=('pi',90))
    
    
    start = accuracy_df.t.min()
    stop = accuracy_df.t.max()
    #sns.set(rc={'figure.figsize':(np.round((stop-start)/20,decimals=1),6)})
    #plt.rcParams["figure.figsize"] = (np.round((stop-start)/30,decimals=1),6)



    #Phase annotation 
    phases = snakemake.config["phase_conditions"].copy()     # TODO only use phases annotation within chosen phase
    phases.pop("all")

    markers = {}
    for phase_name, phase_timings in phases.items():
        markers[phase_timings["start"]] = markers.get(phase_timings["start"], None)
        markers[phase_timings["stop"]] = phase_name

    trans = accuracy_over_time_plot.get_xaxis_transform()
    prev_marker = None
    ordered_marker_keys = np.sort(list(markers.keys()))
    for marker in ordered_marker_keys:
        if marker in range(start-2,stop+2):
            plt.plot([marker,marker], [0,1], ':k', alpha=0.5)
            accuracy_over_time_plot.plot([marker,marker], [1,1.05], color="k", transform=trans, clip_on=False,alpha=0.5)

            if markers[marker] is not None and prev_marker is not None:
                plt.text(0.5 * (marker+prev_marker),1.015,markers[marker],fontsize=11*font_scale,ha='center')
            prev_marker = marker

    # limit range
    accuracy_over_time_plot.set(ylim=[0,1])

    #accuracy_over_time_plot.set(xlim=[0,timepoints_n]) #only shows x range for datapoints
    x_lim = np.max(np.asarray(list(markers.keys())+[timepoints_n],dtype=int))
    #accuracy_over_time_plot.set(xlim=[0,x_lim]) #extends x range for phases TODO integrate better

    #Scale & Shift axis TODO remove hardcoding
    framerate = 15# 30 #15
    stim_start= 30  #30 67,5


    xticks = np.arange(start,stop+1,framerate/2) #tick every second (framerate)
    xtick_labels =xticks-stim_start #scale ticks to second (1/framerate)
    #print(xtick_labels)
    if True: #Change to seconds
        xtick_labels =  np.round(np.divide(xtick_labels,framerate),decimals=1)
        xtick_labels =  ["%.1f" % number for number in xtick_labels]
        logger.info(f"{xtick_labels}")
        plt.xlabel("Seconds")
        

    plt.xticks(xticks, xtick_labels)
    accuracy_over_time_plot.set_xticklabels(xtick_labels)
    #accuracy_over_time_plot.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    

    middle_plot = False
    if middle_plot:
        sns.despine(ax=accuracy_over_time_plot,left=True,right=True)
        accuracy_over_time_plot.set(yticklabels=[])  
        accuracy_over_time_plot.set(ylabel=None)
        accuracy_over_time_plot.tick_params(left=False)
    else:
        plt.ylabel("Accuracy")
    
    #remove vertical lines,
    accuracy_over_time_plot.grid(False) 
    accuracy_over_time_plot.yaxis.grid(True)

    #Random chance
    plt.plot([-.5+start, stop +.5], [1/len(conditions), 1/len(conditions)], '--k') #TODO adapt better
    plt.text(stop-5,1/len(conditions)-0.1,'random chance',fontsize=11*font_scale,ha='center')
    

    #accuracy_over_time_plot.get_figure().show()
    accuracy_over_time_plot.get_figure().savefig( snakemake.output[0] )

    #with open(snakemake.output[1], 'wb') as f:
    #    pickle.dump(accuracy_over_time_plot.get_figure(), f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)

