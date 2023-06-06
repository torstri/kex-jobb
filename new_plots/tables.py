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


def get_abcd(fil, filename):
    temp_freqs = fil[filename]
    abcd = 0
    sift = 0
    for index in temp_freqs:
        if int(index) > 149:
            abcd +=temp_freqs.get(index)
        else:
            sift += temp_freqs.get(index)
    return abcd, sift


def get_frequencies (fil, filename, frequencies):
    temp_freqs = fil[filename]
    print(temp_freqs)
    for index in temp_freqs:
        # print("Index =", index, "Value =", temp_freqs.get(index))
        frequencies[int(index)] += temp_freqs.get(index)
    return frequencies








knn_abcd, knn_sift = get_abcd(knn_sfs, "knn_freqs")
temp_abcd, temp_sift = get_abcd(knn_sbs, "knn_freqs")
knn_abcd += temp_abcd
knn_sift += temp_sift

print("############################")
print("ABCD Knn =", knn_abcd,"And ",float(knn_abcd)/float(19), "SVM Sift =", knn_sift, "And", float(knn_sift)/float(169-19))
print("############################")



nn_abcd, nn_sift = get_abcd(nn_sfs, "nn_freqs")
temp_abcd, temp_sift = get_abcd(nn_sbs, "nn_freqs")
nn_abcd += temp_abcd
nn_sift += temp_sift

print("############################")
print("ABCD nn =", nn_abcd,"And ",float(nn_abcd)/float(19), "SVM Sift =", nn_sift, "And", float(nn_sift)/float(169-19))
print("############################")


rf_abcd, rf_sift = get_abcd(rf_sfs, "rf_freqs")
temp_abcd, temp_sift = get_abcd(rf_sbs, "rf_freqs")
rf_abcd += temp_abcd
rf_sift += temp_sift

print("############################")
print("ABCD rf =", rf_abcd,"And ",float(rf_abcd)/float(19), "SVM Sift =", rf_sift, "And", float(rf_sift)/float(169-19))
print("############################")





svm_abcd, svm_sift = get_abcd(svm_sfs, "svm_freqs")
temp_abcd, temp_sift = get_abcd(svm_sbs, "svm_freqs")
svm_abcd += temp_abcd
svm_sift += temp_sift

print("############################")
print("ABCD SVM =", svm_abcd,"And ",float(svm_abcd)/float(19), "SVM Sift =", svm_sift, "And", float(svm_sift)/float(169-19))
print("############################")

print("All =", svm_abcd + svm_sift + knn_abcd + knn_sift + nn_abcd + nn_sift + rf_abcd + rf_sift)




frequencies = {}
for i in range(0,169):
    frequencies[i] = 0

# print("Frequencies =", frequencies)




frequencies = get_frequencies(knn_sfs, "knn_freqs", frequencies)
frequencies = get_frequencies(knn_sbs, "knn_freqs", frequencies)

frequencies = get_frequencies(nn_sfs, "nn_freqs", frequencies)
frequencies = get_frequencies(nn_sbs, "nn_freqs", frequencies)

frequencies = get_frequencies(rf_sfs, "rf_freqs", frequencies)
frequencies = get_frequencies(rf_sbs, "rf_freqs", frequencies)

frequencies = get_frequencies(svm_sfs, "svm_freqs", frequencies)
frequencies = get_frequencies(svm_sbs, "svm_freqs", frequencies)

print("Frequencies =", frequencies)

sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x:x[1], reverse=True))
abcd  = 0
sift = 0
for feature in frequencies:
    if(int(feature) > 149):
        abcd += frequencies.get(feature)
    else:
        sift += frequencies.get(feature)

print("#################### Table 4.1 ##############################")
print("ABCD =", abcd, "Sift =", sift)
rep_abcd = float(abcd)/float(169 - 148)
rep_sift = float(sift)/float(148)
print("ABCD rep =", rep_abcd, "SIFT rep =", rep_sift)

# Plot sorted frequencies with a bar chart
# plt.bar(range(len(sorted_frequencies)), list(sorted_frequencies.values()), align='center')

# if key in sorted_frequencies is > 149, then it is an ABCD feature. Else, it is a SIFT feature. Make the bar color red if ABCD, blue if SIFT. Keep the current order of the features.
i = 0
for key in sorted_frequencies:
    if(int(key) > 149):
        plt.bar(i, sorted_frequencies.get(key), color='r')
    else:
        plt.bar(i, sorted_frequencies.get(key), color='y')
    i += 1
# sort the values in the bar chart, but keep the color
plt.xticks([])
plt.xlabel('Feature Index')
plt.ylabel('Frequency')
plt.title('Feature Frequency')
# plt.savefig('feature_frequency.png')
plt.show()



