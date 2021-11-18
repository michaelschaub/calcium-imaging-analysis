import numpy as np


import matplotlib.pyplot as plt
#from ipywidgets import interact, IntSlider
from scipy.stats import wilcoxon, stats
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.discriminant_analysis as skda
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
import sklearn.ensemble as skens
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from enum import Enum

from pathlib import Path
import sys
sys.path.append(Path(__file__).parent.parent)
from extract_trials import load_trials, first_frams, extract_all_trials

##### plot through of trial-frames in in pixels space
def plot_through_frames(datainframes, vmin=-0.0009, vmax=0.0009):
    def _plot(index=0):
        image = datainframes[index]
        plt.figure(figsize=(6, 8))
        plt.title('across frames')
        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.colorbar(orientation='horizontal')
        plt.show()

    interact(_plot,
             index=IntSlider(0, 0, datainframes.shape[0] - 1, 1,
                             continuous_update=True))


##### plot average of trial-frames in in pixels space
def plot_frame_average(datainframes, vmin=-0.0009, vmax=0.0009):
    plt.figure(figsize=(6, 8))

    plt.subplot(3, 1, (1, 2))
    mean_image = np.mean(datainframes, axis=0)
    plt.title('mean')
    plt.imshow(mean_image, vmin=vmin, vmax=vmax)
    plt.subplot(3, 1, 3)
    mean_ts = np.mean(datainframes, axis=(1, 2))
    plt.title('time series of spatial mean')
    plt.xlabel('frame')
    plt.ylabel('intensity')
    plt.plot(mean_ts)
    plt.xlim([0, len(mean_ts)])

    plt.tight_layout()
    plt.show()

###### plot AC through different trials
def plot_ac(data):

    def _plot(x0):
        fig = plt.figure()
        fig.set_size_inches(7, 7)
        ACov = data[x0, 0, :, :]
        plt.imshow(ACov, vmin=-0.5, vmax=0.5)
        plt.colorbar(orientation='horizontal')
    interact(_plot, x0=IntSlider(0, 0, data.shape[0] - 1, 1, continuous_update=True))

def colored_violinplot( *args, color=None, facecolor=None, edgecolor=None, **kwargs ):
    violin_parts = plt.violinplot( *args, **kwargs )
    for part in violin_parts:
        #for pc in violin_parts['bodies']:
        parts = violin_parts[part] if part == 'bodies' else [violin_parts[part]]
        for pc in parts:
            if color is not None:
                pc.set_color(color)
            if facecolor is not None:
                pc.set_facecolor(facecolor)
            if edgecolor is not None:
                pc.set_edgecolor(edgecolor)
    return violin_parts


def divide(cor, runs):

    for i_run in range(runs):
        cor[i_run, :, :] /= np.sqrt(np.outer(cor[i_run, :, :].diagonal(),
                                             cor[i_run, :, :].diagonal()))
    return cor


def make_data_with_labels(data):

    datas = np.concatenate(list(data.values()), axis=0)
    labels = list()
    for label in data:
        labels.append(np.full(data[label].shape[0], label.value))
    labels = np.concatenate(labels, axis=0)
    return datas, labels

##### convert from 3D to 2D
def prepare_data(data):
    nsamples, nx, ny = data.shape
    d2_train_dataset = data.reshape((nsamples, nx * ny))
    return d2_train_dataset

##### compute AC
def ac_data(data, comp, n_tau=3, T=7):
    runs = data.shape[0]
    CM = np.zeros([runs, n_tau,comp,comp])
    for run in range(runs):
        for i_tau in range(n_tau):
            CM[run,i_tau,:,:] = np.tensordot(data[run,0:T-n_tau+1,:comp],  data[run,i_tau:T-n_tau+1+i_tau,:comp],
                                             axes=(0,0))/float(T-n_tau)
    return CM


def apply_baseline(mode, data, limit):
    if mode == None:
        data = data
    else:
        data = data[:, limit:, :]
        mean = np.mean(data[:, :limit, :], axis=1, keepdims=True)
        if mode == 'mean':
            def fun(d, m):
                d -= m
        elif mode == 'zscore':
            def fun(d, m):
                d -= m
                d /= np.std(data[:, :limit, :], axis=1, keepdims=True)
        fun(data, mean)

    return data

def calc_features( cond, trials, frames, runs, max_tau=3, max_comps=50, t_interval=7 ):

    ##### cut trials lenghth till the given frame
    data = first_frams(trials[cond], frames)[:runs]

    ##### calculate the mean of the conditions accross frames
    mean = data.mean(1)[:,:max_comps]

    ##### compute the covariance matrix; last parameter is number of components to use,for LDA only 50 would work
    CA = ac_data(data, max_comps, max_tau, t_interval )
    corr = np.copy(CA)[:, 0, :, :]
    ##### divide by square of the outer product of the diagonals
    corr = divide(corr, runs)
    return data[:, :, :max_comps], mean, CA, corr

class Features(Enum):
    DIRECT      = 0
    MEANS       = 1
    COVS        = 2
    TIMECOVS    = 3
    CORRS       = 4


class Conditions(Enum):
    CUES_RIGHT_VIS  = 0
    CUES_LEFT_VIS   = 1
    CUES_RIGHT_TACT = 2
    CUES_LEFT_TACT  = 3

CONDITION_NAMES = {
    Conditions.CUES_RIGHT_VIS  : "cues_right_vis",
    Conditions.CUES_LEFT_VIS   : "cues_left_vis",
    Conditions.CUES_RIGHT_TACT : "cues_right_tact",
    Conditions.CUES_LEFT_TACT  : "cues_left_tact",
}


########################################################################################################
if __name__ == '__main__':

    mouse_ids = ['GN06']
    # relative to current working directory
    root_paths = [ Path.cwd() / Path('data/GN06/2021-01-20_10-15-16') ]
    # relative to script
    root_paths = [ Path(__file__).parent.parent.parent / Path('data/GN06/2021-01-20_10-15-16') ]
    conds = ["cues_right_vis", "cues_left_vis", "cues_left_tact", "cues_right_tact"]
   #################################################
    extract_trials = False #True
    samplingfreq = 15

    if extract_trials:
        trials = extract_all_trials(conds, samplingfreq, root_paths, mouse_ids, run_extraction=True,
                                    save=True)  # Extraction true for first time
    else:
        trials = load_trials(root_paths[0])
    #################################################

    save_outputs = True ##### of classification

    frames = 30 #### trial duration
    baseline_mode = None  #### basline mode ('mean' / 'zscore' / None)
    runs = 200   ### number of runs
    comp = 50   ### number componants to use
    n_rep = 20  ### number of repetition
    n_comp_LDA = 1 ### number of LDA componants (conds -1)

    features = [ Features.MEANS ] #Features.DIRECT, Features.MEANS, Features.COVS, Features.CORRS, Features.TIMECOVS ]
    compared_conds = [ Conditions.CUES_RIGHT_VIS, Conditions.CUES_LEFT_TACT ]

    feature_data = { f : {} for f in features }

    for i,cond in enumerate(compared_conds):
        # calculate all features of trials with condition: cond
        raw, mean, CA, corr = calc_features( CONDITION_NAMES[cond], trials, frames, runs, max_comps=comp )

        for feat in feature_data:
            if feat == Features.DIRECT:
                feature_data[feat][cond] = raw.reshape( runs, -1 )

            if feat == Features.MEANS:
                feature_data[feat][cond] = mean

            if feat in [ Features.CORRS, Features.COVS ]:
                data = corr if feat==Features.CORRS else CA[:,0,:,:]
                # flatten covariances or correlations (discard lower triangle)
                ind = np.triu_indices(data.shape[1], k=0)
                feature_data[feat][cond] = data[ :, ind[0], ind[1] ]

            if feat == Features.TIMECOVS:
                # flatten time lagged covariances
                feature_data[feat][cond] = CA.reshape( runs, -1 )

    feature_labels = {}
    for feat, data in feature_data.items():
        # concat data and create associated labels
        feature_data[feat], feature_labels[feat] = make_data_with_labels( data )
        assert len(feature_data[feat].shape) == 2 and feature_data[feat].shape[0] == feature_labels[feat].shape[0]
        print(feat)
        print(feature_data[feat].shape)


    cv = StratifiedShuffleSplit(n_rep, test_size=0.2, random_state=420)
    perf = np.zeros([n_rep, len(features), 4])
    for i_feat, feat in enumerate(features):
        print(feat.name)

        data = feature_data[feat]
        labels = feature_labels[feat]

        scaler = preprocessing.StandardScaler().fit( data )
        data = scaler.transform(data)

        cv_split = cv.split(data, labels)

        c_MLR = skppl.make_pipeline(skppc.StandardScaler(),
                                    skllm.LogisticRegression(C=1, penalty='l2', multi_class='multinomial',
                                                             solver='lbfgs',
                                                             max_iter=500))
        c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

        c_LDA = skda.LinearDiscriminantAnalysis(n_components=n_comp_LDA, solver='eigen', shrinkage='auto')

        c_RF = skens.RandomForestClassifier(n_estimators=100, bootstrap=True)

        i = 0  ## counter
        for train_idx, test_idx in cv_split:
            print(f'\tRepetition {i:>3}/{n_rep}', end="\r" )
            c_MLR.fit(data[train_idx, :], labels[train_idx])
            c_1NN.fit(data[train_idx, :], labels[train_idx])
            c_LDA.fit(data[train_idx, :], labels[train_idx])
            c_RF .fit(data[train_idx, :], labels[train_idx])
            perf[i, i_feat, 0] = c_MLR.score(data[test_idx, :], labels[test_idx])
            perf[i, i_feat, 1] = c_1NN.score(data[test_idx, :], labels[test_idx])
            perf[i, i_feat, 2] = c_LDA.score(data[test_idx, :], labels[test_idx])
            perf[i, i_feat, 3] = c_RF.score(data[test_idx, :], labels[test_idx])
            i += 1
        print(f'\tRepetition {n_rep}/{n_rep}' )

    if save_outputs:
        np.save('../perf_tasks.npy', perf)
    plt.figure()
    for i, feat in enumerate(features):
        v1 = colored_violinplot(perf[:, i, 0], positions=np.arange(1) + i - 0.3, widths=[0.15], color="blue")
        v2 = colored_violinplot(perf[:, i, 1], positions=np.arange(1) + i - 0.1, widths=[0.15], color="orange")
        v3 = colored_violinplot(perf[:, i, 2], positions=np.arange(1) + i + 0.1, widths=[0.15], color="green")
        v4 = colored_violinplot(perf[:, i, 3], positions=np.arange(1) + i + 0.3, widths=[0.15], color="yellow")
        if i == 0:
            plt.legend( [ v['bodies'][0] for v in [v1,v2,v3,v4]], [ "MLR", "1NN", "LDA","RF" ] )

    plt.xticks(range(len(features)), [ feat.name for feat in features ])
    plt.plot([-.5, len(features)-.5], [0.5, 0.5], '--k')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.savefig("perf_tasks.png")
    plt.show()