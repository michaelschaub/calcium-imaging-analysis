import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import yaml

import matplotlib.ticker as ticker
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

    perf = []

    for path in snakemake.input["perf"]:
        with open(path, "rb") as f:
            perf.append(pickle.load(f))
            
    #print(snakemake.input["perf"])
    #print(perf)

    # config = []
    # for path in snakemake.input["config"]:
    #     with open(path, "r") as f:
    #         config.append(yaml.safe_load(f))

    decoders = [ dec.split('_')[0] for dec in snakemake.params['decoders']]
    conditions = snakemake.params['conds']
    parcellation = None

    perf = np.asarray(perf)
    
    _, timepoints_n, runs_n = perf.shape

    if 'parcellations' in snakemake.params.keys():
        parcellation = snakemake.params['parcellations'] 
        
        
        perf = perf.reshape(len(decoders),len(parcellation), timepoints_n, runs_n)
    
        

    '''
    fig=plt.figure(figsize=[20,10])
    plt.suptitle("Decoding performance")
    violin_plts = []
    colors = cm.get_cmap('Accent',len(decoders)) #[np.arange(0,len(decoders),1)]
    legend = []

    for i,decoder in enumerate(decoders):
        timesteps = len(perf[i])
        width = 0.8 #/(timesteps+2)

        for j,timestep_perf in enumerate(perf[i]):
            #flat_perfs =  np.array(perf[i]).flatten() #list(numpy.concatenate(perf[i]).flat) #Had dimension timepoints x reps
            pos = np.arange(1) + (j) #*1/(timesteps+1))-0.5
            violin_plts.append(plots.colored_violinplot(timestep_perf, positions= pos , widths=[width], color=colors(i/len(decoders))))
        
        legend.append(violin_plts[-1]['bodies'][0])

    plt.legend( legend, decoders,loc='lower left')
    plt.plot([-.5, timesteps -.5], [1/len(conditions), 1/len(conditions)], '--k')
    plt.text(timesteps-0.8,1/len(conditions)+0.01,'random chance',fontsize=7,ha='center')

    offset = -30 #starting point of timepoints relative to t0
   
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(np.arange(0, 1+timesteps, 7.5),np.arange(offset, 1+timesteps+offset, 7.5)) #,np.arange(0, 1+timesteps, 7.5)*(1/15))
    plt.xlabel('Frames (15 Hertz)')

    ax = plt.gca()
    ax.set_ylim([0, 1.05])
    '''
   

    ##### SNS lineplot

    #construct dataframe

    #perf = np.swapaxes(np.asarray(perf),axis1=0, axis2=1)

    #formatted_data = [[decoder, r, timepoint_decoder_run_perf] for r, timepoint_decoder_run_perf in enumerate(timepoint_decoder_perf) for i,timepoint_decoder_perf in enumerate(timepoint_perf) for t,timepoint_perf in enumerate(perf)] 

    #print(formatted_data)

    # data = {'index': [timepoints] ,
    #     'columns': [(decoder, r, run_perf) for r, timepoint_decoder_run_perf in  enumerate(timepoint_decoder_perf) for i,timepoint_decoder_perf in enumerate(timepoint_perf) for t,timepoint_perf in enumerate(perf)] ,
    #     'data': perf,
    #     'index_names': ['timepoints'],
    #     'column_names': ['decoder', 'run', 'accuracy']}
    sns.set(rc={'figure.figsize':(15,7.5)})

    if parcellation is None:
        #Create list of dicts
        accuracy_dict = np.asarray([[[{"decoder": decoder, "t": t, "run":r, "accuracy":run_perf} for r, run_perf in  enumerate(timepoint_perf)] for t,timepoint_perf in enumerate(perf[i,:,:])] for i,decoder in enumerate(decoders)]).flatten()
        accuracy_df =  pd.json_normalize(accuracy_dict)
        accuracy_over_time_plot  = sns.lineplot(data=accuracy_df, x="t", y="accuracy",hue="decoder",errorbar=('pi',90))
    else:
        #Create list of dicts 
        accuracy_dict = np.asarray([[[[{"decoder": decoder, "t": t, "run":r, "accuracy":run_perf, "parcellation":parcel} for r, run_perf in  enumerate(timepoint_perf)] for t,timepoint_perf in enumerate(perf[i,p,:,:])] for i,decoder in enumerate(decoders)] for p, parcel in enumerate(parcellation)]).flatten()
        accuracy_df =  pd.json_normalize(accuracy_dict)

        hue=accuracy_df[['parcellation', 'decoder']].apply(lambda row: f"{row.parcellation}{' (Shuffle)' if 'shuffle' in row.decoder else ''}", axis=1)
        hue.name = 'Parcellation'
        accuracy_over_time_plot  = sns.lineplot(data=accuracy_df, x="t", y="accuracy",hue=hue,errorbar=('pi',90))
    #accuracy_dict = {decoder: {t: timestep for t,timestep in enumerate(perf[i])} for i,decoder in enumerate(decoders)}

     #,orient='index',columns=["accuracy"])

    
    #accuracy_df.set_index("timepoints")
    
    #accuracy_df.index.name="timepoints"


    #print(accuracy_df)

    ### Plotting Seaborn
   

    
     #pi percentile, ci confidence 
    #accuracy_over_time_plot  = sns.lineplot(data=accuracy_df) #, hue="decoder")


    #Phase annotation TODO only use phases annotation within chosen phase
    phases = snakemake.config["phase_conditions"].copy()
    phases.pop("all")

    markers = {}
    for phase_name, phase_timings in phases.items():
        markers[phase_timings["start"]] = markers.get(phase_timings["start"], None)
        markers[phase_timings["stop"]] = phase_name
    print(markers)
    '''
    markers = {
        30: None,
        75: "stimuli",
        82.5: "delay",
        112.5: "response"
    }
    '''
    trans = accuracy_over_time_plot.get_xaxis_transform()

    ordered_marker_keys = np.sort(list(markers.keys()))
    for marker in ordered_marker_keys:


        plt.plot([marker,marker], [0,1], ':k', alpha=0.5)

        #accuracy_over_time_plot.annotate('Neonatal', xy=(1, -.1), xycoords=trans, ha="center", va="top")
        accuracy_over_time_plot.plot([marker,marker], [1,1.05], color="k", transform=trans, clip_on=False,alpha=0.5)

        if markers[marker] is not None:
            plt.text(0.5 * (marker+prev_marker),1.015,markers[marker],fontsize=11,ha='center')


        prev_marker = marker


    # limit range
    accuracy_over_time_plot.set(ylim=[0,1])

    #accuracy_over_time_plot.set(xlim=[0,timepoints_n]) #only shows x range for datapoints
    x_lim = np.max(np.asarray(list(markers.keys())+[timepoints_n],dtype=int))
    accuracy_over_time_plot.set(xlim=[0,x_lim]) #extends x range for phases TODO integrate better

    #Scale & Shift axis TODO remove hardcoding
    framerate = 15# 30 #15
    stim_start= 30  #30 67,5

    xticks = list(range(0,x_lim,framerate)) #tick every second (framerate)
    xtick_labels =np.asarray(xticks)-stim_start #scale ticks to second (1/framerate)
    if True: #Change to seconds
        xtick_labels =  np.asarray(xtick_labels/framerate,dtype=int)
        plt.xlabel("Seconds")

    plt.xticks(xticks, xtick_labels)
    plt.ylabel("Accuracy")
    



    #Random chance
    plt.plot([-.5, timepoints_n -.5], [1/len(conditions), 1/len(conditions)], '--k') #TODO adapt better
    plt.text(timepoints_n+9,1/len(conditions)-0.005,'random chance',fontsize=11,ha='center')

    accuracy_over_time_plot.get_figure().show()
    accuracy_over_time_plot.get_figure().savefig( snakemake.output[0] )

    with open(snakemake.output[1], 'wb') as f:
        pickle.dump(accuracy_over_time_plot.get_figure(), f)

    snakemake_tools.stop_timer(timer_start, logger=logger)
except Exception:
    logger.exception('')
    sys.exit(1)
