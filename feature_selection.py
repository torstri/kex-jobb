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

def test_model(model_function):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2)  # Too much training data for sfs
    
    model = model_function
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred) # initial accuracy
    return accuracy
    
def tests(model_function, forwards, num_features, num_tests):
   
   
    old_accuracy = test_model(model_function) # Get initial accuracy
    
    accuracy = 0 # max accuracy for feature selection
    freqs = {k: 0 for k in range(170)} # Dictionary for feature frequency
    number_of_features = 0
    # Perform tests
    i = 1
    while i <= num_tests:
        
        # Perform feature selection
        sfs_selector = perform_feature_selection(model_function, forwards, num_features)
        
        # Get best accuracy and important features
        features, acc,  = get_important_features(sfs_selector.subsets_,sfs_selector.subsets_[1]['avg_score'], sfs_selector.subsets_[1]['feature_idx'])
        
        # Update frequencies
        for feature in features:
            freqs[feature] += 1
        i +=1
        accuracy += acc # and accuracy
        number_of_features += len(features)

    accuracy = accuracy / num_tests      
    number_of_features = number_of_features / num_tests  
    sorted_freqs = dict(sorted(freqs.items(), key=lambda x:x[1], reverse = True)) # sort frequency list
    return sorted_freqs, accuracy, old_accuracy, number_of_features

    
    
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



num_feats = 10
num_tests = 2

print("--------- S V C ---------")
now = timer()
sorted_freqs, accuracy, old_accuracy, avg_num_features = tests(svm.SVC(C=100), True, num_feats, num_tests)
time = timer() - now

sorted_freqs["avg_features"] = avg_num_features
sorted_freqs["accuracy"]= accuracy
sorted_freqs["initiaL_accuracy"] = old_accuracy
sorted_freqs["cum_time"] = time
print(sorted_freqs)
with open("svm.json", "w") as outfile:
    json.dump(sorted_freqs, outfile)




# perform_and_test_feature_selection(svm.SVC(C=100))
# Takes a very long time if n = 100 (default).
print("\n \n --------- R F ---------")
now = timer()
sorted_freqs, accuracy, old_accuracy, avg_num_features = tests(RandomForestClassifier(n_estimators=5), True, num_feats, num_tests)
sorted_freqs["accuracy"]= accuracy
sorted_freqs["initiaL_accuracy"] = old_accuracy
time = timer() - now
sorted_freqs["cum_time"] = time
print(sorted_freqs)
with open("rf.json", "w") as outfile:
    json.dump(sorted_freqs, outfile)



    
print("\n \n --------- K N N ---------")
now = timer()
sorted_freqs, accuracy, old_accuracy, avg_num_features = tests(KNeighborsClassifier(n_neighbors=8), True, num_feats, num_tests)
time = timer() - now
sorted_freqs["accuracy"]= accuracy
sorted_freqs["initiaL_accuracy"] = old_accuracy
sorted_freqs["cum_time"] = time
print(sorted_freqs)
with open("knn.json", "w") as outfile:
    json.dump(sorted_freqs, outfile)
    




# print("\n \n --------- NEURAL NETWORK---------")
# sorted_freqs = tests(MLPClassifier(
#     hidden_layer_sizes=[8], activation='relu', max_iter=100), True, 2, 2)
# with open("nn.json", "w") as outfile:
#     json.dump(sorted_freqs, outfile)


