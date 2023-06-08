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

# Table 4.3
print("####################### Table 4.3 #######################")

knn_sfs_frequencies = knn_sfs["knn_freqs"]
knn_sbs_frequencies = knn_sbs["knn_freqs"]

nn_sfs_frequencies = nn_sfs["nn_freqs"]
nn_sbs_frequencies = nn_sbs["nn_freqs"]

rf_sfs_frequencies = rf_sfs["rf_freqs"]
rf_sbs_frequencies = rf_sbs["rf_freqs"]


svm_sfs_frequencies = svm_sfs["svm_freqs"]
svm_sbs_frequencies = svm_sbs["svm_freqs"]



for key in knn_sfs_frequencies:
    if(key != 169):
        knn_sfs_frequencies[key] = float(knn_sfs_frequencies.get(key)) /5
        knn_sbs_frequencies[key] = float(knn_sbs_frequencies.get(key)) /5
        
        nn_sfs_frequencies[key] = float(nn_sfs_frequencies.get(key)) /5
        nn_sbs_frequencies[key] = float(nn_sbs_frequencies.get(key)) /5
        
        rf_sfs_frequencies[key] = float(rf_sfs_frequencies.get(key)) /5
        rf_sbs_frequencies[key] = float(rf_sbs_frequencies.get(key)) /5
        
        svm_sfs_frequencies[key] = float(svm_sfs_frequencies.get(key)) /5
        svm_sbs_frequencies[key] = float(svm_sbs_frequencies.get(key)) /5
        
knn_total_frequencies = {}
nn_total_frequencies = {}
rf_total_frequencies = {}
svm_total_frequencies = {}



for i in knn_sbs_frequencies:
    knn_total_frequencies[i] = (knn_sfs_frequencies.get(i) + knn_sbs_frequencies.get(i)) / 2
    nn_total_frequencies[i] = (nn_sfs_frequencies.get(i) + nn_sbs_frequencies.get(i)) / 2
    rf_total_frequencies[i] = (rf_sfs_frequencies.get(i) + rf_sbs_frequencies.get(i)) / 2
    svm_total_frequencies[i] = (svm_sfs_frequencies.get(i) + svm_sbs_frequencies.get(i)) / 2


knn_total_frequencies = dict(sorted(knn_total_frequencies.items(), key=lambda x:x[1], reverse=True))
nn_total_frequencies = dict(sorted(nn_total_frequencies.items(), key=lambda x:x[1], reverse=True))
rf_total_frequencies = dict(sorted(rf_total_frequencies.items(), key=lambda x:x[1], reverse=True))
svm_total_frequencies = dict(sorted(svm_total_frequencies.items(), key=lambda x:x[1], reverse=True))

csv_knn = open('./csv_files/knn.csv', 'w')
knn_write = csv.writer(csv_knn)

csv_nn = open('./csv_files/nn.csv', 'w')
nn_write = csv.writer(csv_nn)

csv_rf = open('./csv_files/rf.csv', 'w')
rf_write = csv.writer(csv_rf)

csv_svm = open('./csv_files/svm.csv', 'w')
svm_write = csv.writer(csv_svm)

header = ['Feature','Total Frequency', 'Frequency using SFS', 'Frequency using SBS']
knn_write.writerow(header)
nn_write.writerow(header)
rf_write.writerow(header)
svm_write.writerow(header)

for index in knn_total_frequencies:
    sfs_value = knn_sfs_frequencies.get(index)
    sbs_value = knn_sbs_frequencies.get(index)
    total = knn_total_frequencies.get(index)
    l = [index, total, sfs_value, sbs_value]
    knn_write.writerows([l])

for index in nn_total_frequencies:
    sfs_value = nn_sfs_frequencies.get(index)
    sbs_value = nn_sbs_frequencies.get(index)
    total = nn_total_frequencies.get(index)
    l = [index, total, sfs_value, sbs_value]
    nn_write.writerows([l])

for index in rf_total_frequencies:
    sfs_value = rf_sfs_frequencies.get(index)
    sbs_value = rf_sbs_frequencies.get(index)
    total = rf_total_frequencies.get(index)
    l = [index, total, sfs_value, sbs_value]
    rf_write.writerows([l])

for index in svm_total_frequencies:
    sfs_value = svm_sfs_frequencies.get(index)
    sbs_value = svm_sbs_frequencies.get(index)
    total = svm_total_frequencies.get(index)
    l = [index, total, sfs_value, sbs_value]
    svm_write.writerows([l])




i = 0
for key in knn_total_frequencies:
    if(int(key) > 149):
        plt.bar(i, knn_total_frequencies.get(key), color='r')
    else:
        plt.bar(i, knn_total_frequencies.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Features sorted in descending order based on frequency')
plt.ylabel('Frequency')
plt.title('Feature Frequency KNN')
# plt.savefig('feature_frequency.png')
plt.show()


i = 0
for key in nn_total_frequencies:
    if(int(key) > 149):
        plt.bar(i, nn_total_frequencies.get(key), color='r')
    else:
        plt.bar(i, nn_total_frequencies.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Features sorted in descending order based on frequency')
plt.ylabel('Frequency')
plt.title('Feature Frequency NN')
# plt.savefig('feature_frequency.png')
plt.show()


i = 0
for key in rf_total_frequencies:
    if(int(key) > 149):
        plt.bar(i, rf_total_frequencies.get(key), color='r')
    else:
        plt.bar(i, rf_total_frequencies.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Features sorted in descending order based on frequency')
plt.ylabel('Frequency')
plt.title('Feature Frequency RF')
# plt.savefig('feature_frequency.png')
plt.show()

i = 0
for key in svm_total_frequencies:
    if(int(key) > 149):
        plt.bar(i, svm_total_frequencies.get(key), color='r')
    else:
        plt.bar(i, svm_total_frequencies.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Features sorted in descending order based on frequency')
plt.ylabel('Frequency')
plt.title('Feature Frequency SVM')
# plt.savefig('feature_frequency.png')
plt.show()

