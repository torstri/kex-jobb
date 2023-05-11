import numpy as np
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

x = np.load('./dataset/balanced/allFeatures.npy')
y = np.load('./SIFT/array_data/y.npy')


def invert_array(indices, total):
    res = np.arange(total)
    return np.delete(res, indices)

# estimator = svm.SVR(kernel="linear", C=100)
# selector = RFE(estimator, n_features_to_select=10, step=1)
# selector = selector.fit(x, y)
# print('support', selector.support_)
# print('ranking', selector.ranking_)

# selector = VarianceThreshold(0.01)
# res = selector.fit_transform(x)
# print(res, " shape: ", res.shape)
# print('time: ', timer() - now)


def perform_and_test_feature_selection(model_function):
    now = timer()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2)  # Too much training data for sfs

    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Old accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))

    model = model_function
    sfs_selector = SequentialFeatureSelector(model, k_features=168,
                                             forward=True,
                                             floating=False,
                                             verbose=2,
                                             scoring='accuracy',
                                             cv=3)
    sfs_selector.fit_transform(x_test, y_test)

    print('Chosen features: ', sfs_selector.k_feature_idx_)
    print("Subsets: ", sfs_selector.subsets_)
    remove = invert_array(sfs_selector.k_feature_idx_, 169)

    fig1 = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')

    plt.ylim([0.6, 1])
    plt.title('Sequential Forward Selection (w. StdDev)')
    plt.grid()
    plt.xticks(range(0, len(sfs_selector.get_metric_dict())+10, 10))
    plt.show()

    x_train = np.delete(x_train, remove, 1)
    x_test = np.delete(x_test, remove, 1)

    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("New accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print('time: ', timer() - now)


print("--------- S V C ---------")
perform_and_test_feature_selection(svm.SVC(C=100))
print("\n \n --------- R F ---------")
# Takes a very long time if n = 100 (default).
perform_and_test_feature_selection(RandomForestClassifier(n_estimators=5))
print("\n \n --------- K N N ---------")
perform_and_test_feature_selection(KNeighborsClassifier(n_neighbors=8))
print("\n \n --------- NEURAL NETWORK---------")
perform_and_test_feature_selection(MLPClassifier(
    hidden_layer_sizes=[8], activation='relu', max_iter=100))
