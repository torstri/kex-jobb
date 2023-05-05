import numpy as np
import sklearn.svm as svm
from sklearn.feature_selection import RFE

x = np.load('./dataset/balanced/allFeatures.npy')
y = np.load('./SIFT/array_data/y.npy')

# svc = SVC(C=100 ,kernel='rbf')
estimator = svm.SVR(kernel="linear", C=100)
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(x, y)
print('support', selector.support_)
print('ranking', selector.ranking_)