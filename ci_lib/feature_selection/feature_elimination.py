import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs


class RFE_pipeline(skppl.Pipeline):
    '''
    MLR adapted for recursive feature elimination (RFE)
    '''
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

def rec_feature_elimination(select_feats_n, class_feats, class_labels, repetitions):
    """
    @param select_feats_n The number of reduced features
    @param class_feats An array that contains the features for each class.
    @param class_labels An array that contains the labels of each class
    @param repetitions The number of repetitions

    @return (The best performing features, the corresponding performance on the test data, the trained models)
    """

    ### Scale & Split
    cv = StratifiedShuffleSplit(repetitions, test_size=0.2, random_state=420)

    data = np.concatenate([feat.flatten() for feat in class_feats])
    labels = np.concatenate([np.full((len(class_feats[i].flatten())), class_labels[i])
                             for i in range(len(class_feats))])


    scaler = preprocessing.StandardScaler().fit( data )
    data = scaler.transform(data)
    cv_split = cv.split(data, labels)

    # Build RFE pipeline
    c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=1.0, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])

    _ , feats = data.shape

    if(float(select_feats_n)<=1):
        select_feats_n= int(np.round(feats*float(select_feats_n)))
    elif(int(select_feats_n)>data.shape[-1]):
        select_feats_n = feats

    RFE = skfs.RFE(c_MLR,n_features_to_select=int(select_feats_n))

    ranking = np.zeros([repetitions,feats],dtype=np.int32)
    perf = np.zeros((repetitions))
    decoders = []

    ### Train & Eval
    for i, (train_i, test_i) in enumerate(cv_split):

        RFE = RFE.fit(data[train_i, :], labels[train_i])
        ranking[i,:] = RFE.ranking_
        best_feat_iter = np.sort(np.argsort(ranking[i])[:int(select_feats_n)])
        perf[i] = RFE.estimator_.score(data[test_i, :][:,best_feat_iter], labels[test_i])
        decoders.append(RFE.estimator_)

    selected_feats = np.argsort(ranking.mean(0))[:int(select_feats_n)]


    return selected_feats, perf, decoders