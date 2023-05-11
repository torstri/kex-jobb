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

def get_important_features(subsets, acc, features):
    highest_accuracy = acc
    # Printa iterationer
    it = 0
    for subset in subsets:
        accuracy = subsets[subset]['avg_score']
        if(highest_accuracy < accuracy):
            highest_accuracy = accuracy
            features = subsets[subset]['feature_idx']
    return features, highest_accuracy

def get_feature_rating(features):
    print("")

def tests(model_function, forwards, num_features, num_tests):
   
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)  # Too much training data for sfs
    
    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    old_accuracy = metrics.accuracy_score(y_test, y_pred) # initial accuracy
    
    accuracies = 0
    # Dictionary for feature frequency
    freqs = {k: 0 for k in range(170)}
    highest_accuracy = 0
    
    i = 1
    while i <= num_tests:
        
        sfs_selector = perform_feature_selection(model_function, forwards, num_features)
        features, accuracy = get_important_features(sfs_selector.subsets_,sfs_selector.subsets_[1]['avg_score'], sfs_selector.subsets_[1]['feature_idx'])
        for feature in features:
            freqs[feature] += 1
        i +=1
        accuracies += accuracy
        
    b = 0
    for feature in freqs:
        if(freqs[feature] > 0):
            print(b)
        b += 1
        
    sorted_freqs = dict(sorted(freqs.items(), key=lambda x:x[1], reverse = True))
    for feature in sorted_freqs:
        print("Feature: ", feature, " Frequency: ", sorted_freqs[feature])
    print("Average accuracy =", accuracies/num_tests)
    print("Initial accuracy = ", old_accuracy)
    
    # # Remove ineffective features
    # remove = invert_array(sfs_selector.k_feature_idx_, 169)
    # x_train = np.delete(x_train, remove, 1)
    # x_test = np.delete(x_test, remove, 1)

    # print("\n================================ Important features ======================================")    
    
    # print("Highest accuracy: ", highest_accuracy)
    # print("Most important features: ", features)
    # print("Number of features: ", len(features))
    # x_train = np.delete(x_train, remove, 1)
    # x_test = np.delete(x_test, remove, 1)
    # print("======================================================================")    
    # print('Chosen features: ', sfs_selector.k_feature_idx_)
    # print("----------------------------------------------------")

    # model = model_function
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # print("Old accuracy: ", old_accuracy)
    # print("New accuracy: ", metrics.accuracy_score(y_test, y_pred))
    # print('time: ', timer() - now)
    # print("ABCD features: ", abcd_features)
    
    
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

def perform_and_test_feature_selection(model_function):
    now = timer()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2)  # Too much training data for sfs

    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    old_accuracy = metrics.accuracy_score(y_test, y_pred)

    model = model_function
    sfs_selector = SequentialFeatureSelector(model, k_features=50,
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
    # Printa iterationer
    # print("\n=================================== Iterations ===================================")
    # it = 0
    # abcd_features = 0
    # for subset in sfs_selector.subsets_:
    #     print('Current feature set: ', sfs_selector.subsets_[subset]['feature_idx'])
    #     diff = 0
    #     if(subset > 1):
            
    #         old = sfs_selector.subsets_[subset - 1]['feature_idx']
    #         new = sfs_selector.subsets_[subset]['feature_idx']
    #         for  new_feature in new:
    #             is_new = True
    #             for old_feature in old:
    #                 if(old_feature == new_feature):
    #                     is_new = False
    #                     break
    #             if(is_new):
    #                 print("New feature = ", new_feature)
    #                 diff = new_feature
        
    #     if(diff > 154):
    #             print("New feature is ABCD")
    #             abcd_features += 1
    #     print('Current accuracy: ', sfs_selector.subsets_[subset]['avg_score'])
    #     print("-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --")    
    #     it+=1
    
    
    features, highest_accuracy = get_important_features(sfs_selector.subsets_,sfs_selector.subsets_[1]['avg_score'], sfs_selector.subsets_[1]['feature_idx'])
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
    print("ABCD features: ", abcd_features)
    # Plott
    # fig1 = plot_sfs(sfs_selector.get_metric_dict(), kind='std_dev')

    # plt.ylim([0.8, 1])
    # plt.title('Sequential Forward Selection (w. StdDev)')
    # plt.grid()
    # plt.show()


print("--------- S V C ---------")
tests(svm.SVC(C=100), True, 5, 5)


# dict = {k: 0 for k in range(170)}

# dict =dict.fromkeys(range(170), 0)
# print("Dict: ", dict)
# dict[0] += 1
# print("Dict ", dict)
# print("=======")
# for feature in dict:
#     print(feature, " ", dict[feature])


# perform_and_test_feature_selection(svm.SVC(C=100))
# print("\n \n --------- R F ---------")
# # Takes a very long time if n = 100 (default).
# perform_and_test_feature_selection(RandomForestClassifier(n_estimators=5))
# print("\n \n --------- K N N ---------")
# perform_and_test_feature_selection(KNeighborsClassifier(n_neighbors=8))
# print("\n \n --------- NEURAL NETWORK---------")
# perform_and_test_feature_selection(MLPClassifier(
#     hidden_layer_sizes=[8], activation='relu', max_iter=100))


