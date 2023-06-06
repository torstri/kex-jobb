import json
from numpy import *
import math
import matplotlib.pyplot as plt
import csv


fp = open('new_knn_SFS.json')
knn_sfs = json.load(fp)
fp = open('new_knn_SBS.json')
knn_sbs = json.load(fp)

fp = open('new_nn_SFS.json')
nn_sfs = json.load(fp)
fp = open('new_nn_SBS.json')
nn_sbs = json.load(fp)

fp = open('new_rf_SFS.json')
rf_sfs = json.load(fp)
fp = open('new_rf_SBS.json')
rf_sbs = json.load(fp)

fp = open('new_svm_SFS.json')
svm_sfs = json.load(fp)
fp = open('new_svm_SBS.json')
svm_sbs = json.load(fp)


def get_feature_frequencies(fil_sfs,fil_sbs, classifier):
    frequencies = {}
    for key in fil_sfs[classifier]:
        frequencies[int(key)] = fil_sfs[classifier].get(key) + fil_sbs[classifier].get(key)
    return frequencies 


def get_average_num_features(fil, classifier_features, classifier_accuracy):
    data = fil[classifier_features]
    first_run = data.get("1")
    second_run = data.get("2")
    third_run = data.get("3")
    fourth_run = data.get("4")
    fifth_run = data.get("5")
    average_features = (first_run + second_run + third_run + fourth_run+ fifth_run)/5
    average_accuracy = fil[classifier_accuracy]
    
    return average_features, average_accuracy



    

# Get average number of features for each model and method
knn_sfs_average_features, knn_sfs_average_accuracy = get_average_num_features(knn_sfs, "knn_num_features", "knn_accuracy")
knn_sbs_average_features, knn_sbs_average_accuracy  = get_average_num_features(knn_sbs, "knn_num_features", "knn_accuracy")

nn_sfs_average_features, nn_sfs_average_accuracy = get_average_num_features(nn_sfs, "nn_num_features", "nn_accuracy")
nn_sbs_average_features, nn_sbs_average_accuracy  = get_average_num_features(nn_sbs, "nn_num_features", "nn_accuracy")

rf_sfs_average_features, rf_sfs_average_accuracy = get_average_num_features(rf_sfs, "rf_num_features", "rf_accuracy")
rf_sbs_average_features, rf_sbs_average_accuracy  = get_average_num_features(rf_sbs, "rf_num_features", "rf_accuracy")

svm_sfs_average_features, svm_sfs_average_accuracy = get_average_num_features(svm_sfs, "svm_num_features", "svm_accuracy")
svm_sbs_average_features, svm_sbs_average_accuracy  = get_average_num_features(svm_sbs, "svm_num_features", "svm_accuracy")



# Open and write files
csv_knn = open('./csv_files/knn_selection_methods.csv', 'w')
knn_write = csv.writer(csv_knn)

csv_nn = open('./csv_files/nn_selection_methods.csv', 'w')
nn_write = csv.writer(csv_nn)

csv_rf = open('./csv_files/rf_selection_methods.csv', 'w')
rf_write = csv.writer(csv_rf)

csv_svm = open('./csv_files/svm_selection_methods.csv', 'w')
svm_write = csv.writer(csv_svm)

# header = ['Classifier','Selection Method', 'Average Number of Features', 'Average Accuracy']
# knn_write.writerow(header)
# knn_write.writerows([["KNN", "SFS", knn_sfs_average_features, knn_sfs_average_accuracy]])
# knn_write.writerows([["KNN", "SFS", knn_sbs_average_features, knn_sbs_average_accuracy]])

# nn_write.writerow(header)
# nn_write.writerows([["NN", "SFS", nn_sfs_average_features, nn_sfs_average_accuracy]])
# nn_write.writerows([["NN", "SFS", nn_sbs_average_features, nn_sbs_average_accuracy]])

# rf_write.writerow(header)
# rf_write.writerows([["RF", "SFS", rf_sfs_average_features, rf_sfs_average_accuracy]])
# rf_write.writerows([["RF", "SFS", rf_sbs_average_features, rf_sbs_average_accuracy]])

# svm_write.writerow(header)
# svm_write.writerows([["SVM", "SFS", svm_sfs_average_features, svm_sfs_average_accuracy]])
# svm_write.writerows([["SVM", "SFS", svm_sbs_average_features, svm_sbs_average_accuracy]])



knn_features = get_feature_frequencies(knn_sfs, knn_sbs, "knn_freqs")
nn_features = get_feature_frequencies(nn_sfs, nn_sbs, "nn_freqs")
rf_features = get_feature_frequencies(rf_sfs, rf_sbs, "rf_freqs")
svm_features = get_feature_frequencies(svm_sfs, svm_sbs, "svm_freqs")

knn_features = dict(sorted(knn_features.items(), key=lambda x:x[1], reverse=True))
nn_features = dict(sorted(nn_features.items(), key=lambda x:x[1], reverse=True))
rf_features = dict(sorted(rf_features.items(), key=lambda x:x[1], reverse=True))
svm_features = dict(sorted(svm_features.items(), key=lambda x:x[1], reverse=True))





# if key in sorted_frequencies is > 149, then it is an ABCD feature. Else, it is a SIFT feature. Make the bar color red if ABCD, blue if SIFT. Keep the current order of the features.
i = 0
for key in knn_features:
    if(int(key) > 149):
        plt.bar(i, knn_features.get(key), color='r')
    else:
        plt.bar(i, knn_features.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Feature Index')
plt.ylabel('Frequency')
plt.title('Feature Frequency for KNN')
# plt.savefig('feature_frequency.png')
plt.show()

i = 0
for key in nn_features:
    if(int(key) > 149):
        plt.bar(i, nn_features.get(key), color='r')
    else:
        plt.bar(i, nn_features.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Feature Index')
plt.ylabel('Frequency')
plt.title('Feature Frequency for NN')
# plt.savefig('feature_frequency.png')
plt.show()

i = 0
for key in rf_features:
    if(int(key) > 149):
        plt.bar(i, rf_features.get(key), color='r')
    else:
        plt.bar(i, rf_features.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Feature Index')
plt.ylabel('Frequency')
plt.title('Feature Frequency for RF')
# plt.savefig('feature_frequency.png')
plt.show()

i = 0
for key in svm_features:
    if(int(key) > 149):
        plt.bar(i, svm_features.get(key), color='r')
    else:
        plt.bar(i, svm_features.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Feature Index')
plt.ylabel('Frequency')
plt.title('Feature Frequency for SVM')
# plt.savefig('feature_frequency.png')
plt.show()