import numpy as np
from extract_trials import load_trials, first_frams, extract_all_trials
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from scipy.stats import wilcoxon, stats
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.discriminant_analysis as skda
import sklearn.preprocessing as skppc
import sklearn.pipeline as skppl
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path


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


def divide(cor, runs):

    for i_run in range(runs):
        cor[i_run, :, :] /= np.sqrt(np.outer(cor[i_run, :, :].diagonal(),
                                             cor[i_run, :, :].diagonal()))
    return cor


def make_data_with_labels(data):

    datas = np.concatenate(data, axis=0)
    labels = list()
    for index, element in enumerate(data):
        labels.append(np.full(element.shape[0], index))
    labs = np.concatenate(labels, axis=0)
    return datas, labs

##### convert from 3D to 2D
def prepare_data(data):
    nsamples, nx, ny = data.shape
    d2_train_dataset = data.reshape((nsamples, nx * ny))
    return d2_train_dataset

##### compute AC
def ac_data(data, runs, comp):
    n_tau = 3
    T = 7
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


########################################################################################################
if __name__ == '__main__':

    mouse_ids = ['GN06']
    # relative to current working directory
    root_paths = [ Path.cwd() / Path('data/GN06/2021-01-20_10-15-16') ]
    # relative to script
    root_paths = [ Path(__file__).parent.parent / Path('data/GN06/2021-01-20_10-15-16') ]
    conds = ["cues_right_vis", "cues_left_vis", "cues_left_tact", "cues_right_tact"]
   #################################################
    extract_trials = False #True
    samplingfreq = 15

    if extract_trials:
        trials = extract_all_trials(conds, samplingfreq, root_paths, mouse_ids, run_extraction=True, save=True) #Extraction true for first time
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


    ##### cut trials lenghth till the given frame
    cond1 = first_frams(trials["cues_right_vis"], frames)[:runs]
    cond2 = first_frams(trials["cues_left_vis"], frames)[:runs]
    cond3 = first_frams(trials["cues_right_tact"], frames)[:runs]
    cond4 = first_frams(trials["cues_left_tact"], frames)[:runs]



    ########################
    # This code is only to look at the data in the pixels space

    # file_path = r'C:\Users\Mo\2021-01-20_10-15-16\SVD_data\Vc.mat'
    # f = h5py.File(file_path, 'r')
    # U = np.array(f['U'])  # mixing matrix (component x voxel x voxel)
    # def bproj(x):
    # return np.tensordot(x, U, axes=(-1, 0))

    # projected_in_pixels_space = bproj(corrected.mean(0))
    # plot_through_frames(projected_in_pixels_space)
    # plot_frame_average(projected_in_pixels_space)

    #########################

    ##### compute the covariance matrix; last parameter is number of components to use,for LDA only 50 would work

    CA1 = ac_data(cond1, runs, comp)
    CA2 = ac_data(cond2, runs, comp)
    CA3 = ac_data(cond3, runs, comp)
    CA4 = ac_data(cond4, runs, comp)

    ####plot when needed
    # plot_ca(CA1)
    # a = CA3[:,0,:,:].flatten()
    # b = CA1[:,0,:,:].flatten()
    # stat, p = wilcoxon(a,b)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))

    corr1 = np.copy(CA1)[:, 0, :, :]
    corr2 = np.copy(CA2)[:, 0, :, :]
    corr3 = np.copy(CA3)[:, 0, :, :]
    corr4 = np.copy(CA4)[:, 0, :, :]

    ##### divide by square of the outer product of the diagonals
    corr1 = divide(corr1, runs)
    corr2 = divide(corr2, runs)
    corr3 = divide(corr3, runs)
    corr4 = divide(corr4, runs)

    ##### Use all the conditions lenght for calssification
    #corrected1 = corrected1.reshape([runs,-1])
    #corrected2 = corrected2.reshape([runs,-1])
    #corrected3 = corrected3.reshape([runs,-1])
    #corrected4 = corrected4.reshape([runs,-1])

    ##### calculate the mean of the conditions accross frames
    cond1 = cond1.mean(1)
    cond2 = cond2.mean(1)
    cond3 = cond3.mean(1)
    cond4 = cond4.mean(1)

    ##### add data for classification to the lists
    means = [cond1, cond4]
    corrs = [corr1, corr4]

    ######## correlation data & labels
    data1, labels1 = make_data_with_labels(corrs)

    ####### from 3D into 2D only for correlation data
    data1 = prepare_data(data1)

    ######## mean data & labels
    data2, labels2 = make_data_with_labels(means)

    ######## (mean & correlation) data & labels
    data3  = np.concatenate([data1,data2], axis= 1)

    n_conn = 3  ### number of measures (mean & correlation & (mean & correlation))

    cv = StratifiedShuffleSplit(n_rep, test_size=0.2, random_state=420)
    perf = np.zeros([n_rep, n_conn, 3])
    for i_conn in range(n_conn):

        if i_conn == 0:
            data, labels = data1, labels1
        elif i_conn == 1:
            data, labels = data2, labels2
        elif i_conn == 2:
            data, labels = data3, labels1

        cv_split = cv.split(data, labels)

        c_MLR = skppl.make_pipeline(skppc.StandardScaler(),
                                    skllm.LogisticRegression(C=1, penalty='l2', multi_class='multinomial',
                                                             solver='lbfgs',
                                                             max_iter=500))
        c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')

        c_LDA = skda.LinearDiscriminantAnalysis(n_components=n_comp_LDA, solver='eigen', shrinkage='auto')

        i = 0  ## counter
        for train_idx, test_idx in cv_split:
            print('\tRepetition %d' % i)
            c_MLR.fit(data[train_idx, :], labels[train_idx])
            c_1NN.fit(data[train_idx, :], labels[train_idx])
            c_LDA.fit(data[train_idx, :], labels[train_idx])
            perf[i, i_conn, 0] = c_MLR.score(data[test_idx, :], labels[test_idx])
            perf[i, i_conn, 1] = c_1NN.score(data[test_idx, :], labels[test_idx])
            perf[i, i_conn, 2] = c_LDA.score(data[test_idx, :], labels[test_idx])
            i += 1

    if save_outputs:
        np.save('perf_tasks.npy', perf)
    plt.figure()
    plt.violinplot(perf[:, 1, 0], positions=np.arange(1) - 0.3, widths=[0.2])
    plt.violinplot(perf[:, 1, 1], positions=np.arange(1), widths=[0.2])
    plt.violinplot(perf[:, 1, 2], positions=np.arange(1) + 0.3, widths=[0.2])

    plt.violinplot(perf[:, 0, 0], positions=np.arange(1) + 0.8, widths=[0.2])
    plt.violinplot(perf[:, 0, 1], positions=np.arange(1) + 1.1, widths=[0.2])
    plt.violinplot(perf[:, 0, 2], positions=np.arange(1) + 1.4, widths=[0.2])

    plt.violinplot(perf[:, 2, 0], positions=np.arange(1) + 1.8, widths=[0.2])
    plt.violinplot(perf[:, 2, 1], positions=np.arange(1) + 2.1, widths=[0.2])
    plt.violinplot(perf[:, 2, 2], positions=np.arange(1) + 2.4, widths=[0.2])

    plt.xticks(range(3), ["Mean", "FC", "Mean + FC"])
    plt.plot([-1, 3], [0.5, 0.5], '--k')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.ylabel('Accuracy', fontsize=14)
    plt.show()
