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



    


knn_sfs_average_features, knn_sfs_average_accuracy = get_average_num_features(knn_sfs, "knn_num_features", "knn_accuracy")
knn_sbs_average_features, knn_sbs_average_accuracy  = get_average_num_features(knn_sbs, "knn_num_features", "knn_accuracy")

nn_sfs_average_features, nn_sfs_average_accuracy = get_average_num_features(nn_sfs, "nn_num_features", "nn_accuracy")
nn_sbs_average_features, nn_sbs_average_accuracy  = get_average_num_features(nn_sbs, "nn_num_features", "nn_accuracy")

rf_sfs_average_features, rf_sfs_average_accuracy = get_average_num_features(rf_sfs, "rf_num_features", "rf_accuracy")
rf_sbs_average_features, rf_sbs_average_accuracy  = get_average_num_features(rf_sbs, "rf_num_features", "rf_accuracy")

svm_sfs_average_features, svm_sfs_average_accuracy = get_average_num_features(svm_sfs, "svm_num_features", "svm_accuracy")
svm_sbs_average_features, svm_sbs_average_accuracy  = get_average_num_features(svm_sbs, "svm_num_features", "svm_accuracy")





print("SVM")
print("Average num features SFS = ", svm_sfs_average_features, "Average accuracy SFS =", svm_sfs_average_accuracy)
print("Average num features SBS = ", svm_sbs_average_features, "Average accuracy SBS =", svm_sbs_average_accuracy)


csv_knn = open('./csv_files/knn_selection_methods.csv', 'w')
knn_write = csv.writer(csv_knn)

csv_nn = open('./csv_files/nn_selection_methods.csv', 'w')
nn_write = csv.writer(csv_nn)

csv_rf = open('./csv_files/rf_selection_methods.csv', 'w')
rf_write = csv.writer(csv_rf)

csv_svm = open('./csv_files/svm_selection_methods.csv', 'w')
svm_write = csv.writer(csv_svm)

header = ['Classifier','Selection Method', 'Average Number of Features', 'Average Accuracy']
knn_write.writerow(header)
knn_write.writerows([["KNN", "SFS", knn_sfs_average_features, knn_sfs_average_accuracy]])
knn_write.writerows([["KNN", "SFS", knn_sbs_average_features, knn_sbs_average_accuracy]])

nn_write.writerow(header)
nn_write.writerows([["NN", "SFS", nn_sfs_average_features, nn_sfs_average_accuracy]])
nn_write.writerows([["NN", "SFS", nn_sbs_average_features, nn_sbs_average_accuracy]])

rf_write.writerow(header)
rf_write.writerows([["RF", "SFS", rf_sfs_average_features, rf_sfs_average_accuracy]])
rf_write.writerows([["RF", "SFS", rf_sbs_average_features, rf_sbs_average_accuracy]])

svm_write.writerow(header)
svm_write.writerows([["SVM", "SFS", svm_sfs_average_features, svm_sfs_average_accuracy]])
svm_write.writerows([["SVM", "SFS", svm_sbs_average_features, svm_sbs_average_accuracy]])
