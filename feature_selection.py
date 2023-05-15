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
import json

from sklearn.model_selection import GridSearchCV

x = np.load('./dataset/balanced/allFeatures.npy')
y = np.load('./SIFT/array_data/y.npy')


def invert_array(indices, total):
    res = np.arange(total)
    return np.delete(res, indices)


def perform_and_test_feature_selection(model_function):
    now = timer()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2)  # Too much training data for sfs

    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    old_accuracy = metrics.accuracy_score(y_test, y_pred)

    model = model_function
    sfs_selector = SequentialFeatureSelector(model, k_features=168,
                                             forward=True,
                                             floating=False,
                                             verbose=2,
                                             scoring='accuracy',
                                             cv=3)
    sfs_selector.fit_transform(x_test, y_test)

    print('Chosen features: ', sfs_selector.k_feature_idx_)
    # print("Subsets: ", sfs_selector.subsets_)
    remove = invert_array(sfs_selector.k_feature_idx_, 169)
    print("---------------------------------------------------")
    # print('Removed features: ', remove)
    highest_accuracy = old_accuracy

    features, _, highest_accuracy = get_important_features(sfs_selector.subsets_, sfs_selector.subsets_[
                                                        1]['avg_score'], sfs_selector.subsets_[1]['feature_idx'])
    print("\n================================ Important features ======================================")

    print("Highest accuracy: ", highest_accuracy)
    print("Most important features: ", features)
    print("Number of features: ", len(features))
    x_train = np.delete(x_train, remove, 1)
    x_test = np.delete(x_test, remove, 1)
    print("======================================================================")
    print('Chosen features: ', sfs_selector.k_feature_idx_)
    print("----------------------------------------------------")

    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Old accuracy: ", old_accuracy)
    print("New accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print('time: ', timer() - now)


# estimator = svm.SVR(kernel="linear", C=100)
# selector = RFE(estimator, n_features_to_select=10, step=1)
# selector = selector.fit(x, y)
# print('support', selector.support_)
# print('ranking', selector.ranking_)

# selector = VarianceThreshold(0.01)
# res = selector.fit_transform(x)
# print(res, " shape: ", res.shape)
# print('time: ', timer() - now)

def get_important_features(subsets, acc, features):
    highest_accuracy = acc
    accurancies = []

    for subset in subsets:
        print("Subset: ", subset)
        accuracy = subsets[subset]['avg_score']
        accurancies.append(accuracy)
        if (highest_accuracy < accuracy):
            highest_accuracy = accuracy
            features = subsets[subset]['feature_idx']

    return features, accurancies, highest_accuracy


def test_model(model_function):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)  # Too much training data for sfs

    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)  # initial accuracy
    return accuracy

# Perform feature selection
# @param model_function: ML model
# @param forwards: SFS if true, SBS else
# @return sfs_selector:


def perform_feature_selection(model_function, forwards, num_features):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)  # Too much training data for sfs

    model = model_function
    sfs_selector = SequentialFeatureSelector(model, k_features=num_features,
                                             forward=forwards,
                                             floating=False,
                                             verbose=2,
                                             scoring='accuracy',
                                             cv=3)
    sfs_selector.fit_transform(x_test, y_test)

    return sfs_selector


def param_search():
    print("")


def tests(model_function, forwards, num_features, num_tests):

    old_accuracy = test_model(model_function)  # Get initial accuracy

    accuracy = 0  # max accuracy for feature selection
    all_accurancies = {k: 0 for k in range(0, 170)}
    freqs = {k: 0 for k in range(170)}  # Dictionary for feature frequency
    number_of_features = {}
    # Perform tests
    i = 1
    index = 1
    if (not forwards):
        index = 169
    while i <= num_tests:
        print("-------------------------------- Test Nr. ",
              i, " --------------------------------")
        # Perform feature selection
        sfs_selector = perform_feature_selection(
            model_function, forwards, num_features)

        # Get best accuracy and important features
        features, accurancies, acc = get_important_features(sfs_selector.subsets_, sfs_selector.subsets_[
                                               index]['avg_score'], sfs_selector.subsets_[index]['feature_idx'])

        # Update frequencies
        for feature in features:
            freqs[feature] += 1
        accuracy += acc  # and accuracy
        print('accuracies: ', accurancies)
        for j in range(0, len(accurancies)):
            all_accurancies[j] = all_accurancies[j] + accurancies[j]
        number_of_features[i] = len(features)
        i += 1

    accuracy = accuracy / num_tests
    # sort frequency list
    sorted_freqs = dict(
        sorted(freqs.items(), key=lambda x: x[1], reverse=True))
    
    for i in range(0, len(all_accurancies)):
        all_accurancies[i] = all_accurancies[i] / num_tests

    return sorted_freqs, accuracy, old_accuracy, number_of_features, all_accurancies


def perform_tests(forwards, num_features, num_tests):
    svm_results = {}
    rf_results = {}
    knn_results = {}
    nn_results = {}

    print("-------------------------------- SVM --------------------------------")
    now = timer()
    svm_freqs, svm_accuracy, svm_old_accuracy, svm_num_features, all_accs = tests(
        svm.SVC(C=100), forwards=forwards, num_features=num_features, num_tests=num_tests)
    time = timer() - now

    svm_results["svm_freqs"] = svm_freqs
    svm_results["svm_accuracy"] = svm_accuracy
    svm_results["svm_old_accuracy"] = svm_old_accuracy
    svm_results["svm_num_features"] = svm_num_features
    svm_results["svm_cum_time"] = time
    svm_results["all_accs"] = all_accs

    print("-------------------------------- RF --------------------------------")
    now = timer()
    rf_freqs, rf_accuracy, rf_old_accuracy, rf_num_features, all_accs = tests(RandomForestClassifier(
        n_estimators=5), forwards=forwards, num_features=num_features, num_tests=num_tests)
    time = timer() - now

    rf_results["rf_freqs"] = rf_freqs
    rf_results["rf_accuracy"] = rf_accuracy
    rf_results["rf_old_accuracy"] = rf_old_accuracy
    rf_results["rf_num_features"] = rf_num_features
    rf_results["rf_cum_time"] = time
    rf_results["all_accs"] = all_accs

    print("-------------------------------- KNN --------------------------------")
    now = timer()
    knn_freqs, knn_accuracy, knn_old_accuracy, knn_num_features, all_accs = tests(KNeighborsClassifier(
        n_neighbors=8), forwards=forwards, num_features=num_features, num_tests=num_tests)
    time = timer() - now

    knn_results["knn_freqs"] = knn_freqs
    knn_results["knn_accuracy"] = knn_accuracy
    knn_results["knn_old_accuracy"] = knn_old_accuracy
    knn_results["knn_num_features"] = knn_num_features
    knn_results["knn_cum_time"] = time
    knn_results["all_accs"] = all_accs

    print("-------------------------------- NEURAL NETWORK --------------------------------")
    now = timer()
    nn_freqs, nn_accuracy, nn_old_accuracy, nn_num_features, all_accs = tests(MLPClassifier(hidden_layer_sizes=[8], activation='relu', max_iter=100), forwards = forwards, num_features = num_features, num_tests = num_tests)
    time=timer() - now

    nn_results["nn_freqs"]=nn_freqs
    nn_results["nn_accuracy"]=nn_accuracy
    nn_results["nn_old_accuracy"]=nn_old_accuracy
    nn_results["nn_num_features"]=nn_num_features
    nn_results["nn_cum_time"]=time
    nn_results["all_accs"] = all_accs

    # Write to json files
    filenames=["svm_SFS.json", "rf_SFS.json", "knn_SFS.json", "nn_SFS.json"]
    if (not forwards):
        filenames=["svm_SBS.json", "rf_SBS.json", "knn_SBS.json", "nn_SBS.json"]

    fp_svm=open(filenames[0], "w")
    json.dump(svm_results, fp_svm)
    fp=open(filenames[1], "w")
    json.dump(rf_results, fp)
    fp=open(filenames[2], "w")
    json.dump(knn_results, fp)
    fp=open(filenames[3], "w")
    json.dump(nn_results, fp)


num_feats=168
num_tests=5

perform_tests(True, num_feats, num_tests)

# Fast test
# perform_tests(True, 5, 2)

# print("--------- S V C ---------")
# now = timer()
# sorted_freqs, accuracy, old_accuracy, avg_num_features = tests(svm.SVC(C=100), True, num_feats, num_tests)
# time = timer() - now

# sorted_freqs["avg_features"] = avg_num_features
# sorted_freqs["accuracy"]= accuracy
# sorted_freqs["initiaL_accuracy"] = old_accuracy
# sorted_freqs["cum_time"] = time
# print(sorted_freqs)
# with open("svm.json", "w") as outfile:
#     json.dump(sorted_freqs, outfile)




# # perform_and_test_feature_selection(svm.SVC(C=100))
# # Takes a very long time if n = 100 (default).
# print("\n \n --------- R F ---------")
# now = timer()
# sorted_freqs, accuracy, old_accuracy, avg_num_features = tests(RandomForestClassifier(n_estimators=5), True, num_feats, num_tests)
# sorted_freqs["accuracy"]= accuracy
# sorted_freqs["initiaL_accuracy"] = old_accuracy
# time = timer() - now
# sorted_freqs["cum_time"] = time
# print(sorted_freqs)
# with open("rf.json", "w") as outfile:
#     json.dump(sorted_freqs, outfile)




# print("\n \n --------- K N N ---------")
# now = timer()
# sorted_freqs, accuracy, old_accuracy, avg_num_features = tests(KNeighborsClassifier(n_neighbors=8), True, num_feats, num_tests)
# time = timer() - now
# sorted_freqs["accuracy"]= accuracy
# sorted_freqs["initiaL_accuracy"] = old_accuracy
# sorted_freqs["cum_time"] = time
# print(sorted_freqs)
# with open("knn.json", "w") as outfile:
#     json.dump(sorted_freqs, outfile)





# print("\n \n --------- NEURAL NETWORK---------")
# sorted_freqs = tests(MLPClassifier(
#     hidden_layer_sizes=[8], activation='relu', max_iter=100), True, 2, 2)
# with open("nn.json", "w") as outfile:
#     json.dump(sorted_freqs, outfile)
