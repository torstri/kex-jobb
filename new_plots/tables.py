import json
from numpy import *
import math
import matplotlib.pyplot as plt


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


frequencies = {}
for i in range(0,169):
    frequencies[i] = 0
print("Freqs =", frequencies)
print("hello")
temp_freqs = knn_sfs["knn_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)
print("hello")

temp_freqs = knn_sbs["knn_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)

print("hello")

temp_freqs = nn_sfs["nn_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)

temp_freqs = nn_sbs["nn_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)


temp_freqs = rf_sfs["rf_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)

temp_freqs = rf_sbs["rf_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)

temp_freqs = svm_sbs["svm_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)

temp_freqs = svm_sfs["svm_freqs"]
for index in temp_freqs:
    frequencies[int(index)] += temp_freqs.get(index)


print("Freqs = ",frequencies)
sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x:x[1], reverse=True))
print("Sorted freqs =", sorted_frequencies)
abcd  = 0
sift = 0
for feature in frequencies:
    print("Feature = ", feature, "Value =", frequencies.get(feature))
    if(int(feature) > 149):
        print("ABCD feature")
        abcd += frequencies.get(feature)
    else:
        print("SIFT feature")
        
        sift += frequencies.get(feature)
print("ABCD =", abcd, "Sift =", sift)
rep_abcd = float(abcd)/float(169 - 148)
rep_sift = float(sift)/float(148)
print("ABCD rep =", rep_abcd, "SIFT rep =", rep_sift)