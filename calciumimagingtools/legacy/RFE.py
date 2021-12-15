import numpy as np
import sklearn.linear_model as skllm
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms


# MLR adapted for recursive feature elimination (RFE)
class RFE_pipeline(skppl.Pipeline):
    def fit(self, X, y=None, **fit_params):
        """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
        """
        super(RFE_pipeline, self).fit(X, y, **fit_params)
        self.coef_ = self.steps[-1][-1].coef_
        return self

#RFE
def RFE(vect_features, labels, n_comps, type_measure=1):
    N = n_comps

    c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=500))])

    # cross-validation scheme
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    n_rep = 10 # number of repetitions

    # RFE wrappers
    RFE_node = skfs.RFE(c_MLR,n_features_to_select=1)
    RFE_inter = skfs.RFE(c_MLR,n_features_to_select=int(N/2))

    # record classification performance
    rk_node = np.zeros([n_rep,N],dtype=np.int) # RFE rankings for node-type measures (N feature)
    rk_inter = np.zeros([n_rep,int(N*(N-1)/2)],dtype=np.int) # RFE rankings for interaction-type measures (N(N-1)/2 feature)

    '''
    if type_measure == 0:
        vect_features = np.random.rand(S,N)
        vect_features[:int(S/2),:] += np.outer(np.ones(int(S/2)), np.arange(N)/N) # bias that differ across features for 1st class (half of samples)
    else:
        vect_features = np.zeros([S,N,N])
        for s in range(S):
            W = np.eye(N)
            if s<S/2:
                W[:int(N/2),:int(N/2)] += np.random.rand(int(N/2),int(N/2))
            ts_tmp = np.dot(W, np.random.rand(N, 500)) # random time series with and without correlations (for 1st half of nodes)
            vect_features[s,:,:] = np.corrcoef(ts_tmp, rowvar=True)
        vect_features = vect_features[:,mask_tri] # retain only a triangle from whole matrix
    '''

    # loop over repetitions (train/test sets)
    for i_rep in range(n_rep):

        for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1

            # RFE for MLR
            if type_measure == 0: # node-wise feature
                RFE_node.fit(vect_features[ind_train,:], labels[ind_train])
                rk_node[i_rep,:] = RFE_node.ranking_
            else: # interaction-wise feature
                RFE_inter.fit(vect_features[ind_train,:], labels[ind_train])
                rk_inter[i_rep,:] = RFE_inter.ranking_
    if type_measure == 0:
        return rk_node
    else:
        return rk_inter