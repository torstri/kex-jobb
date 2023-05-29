
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


def plot_graph(sfs, sbs):
    sfs_averages = [0]*168
    sbs_averages = [0]*168
    
    for i in range(0,168):
        sfs_averages[i] = (sfs["0"][i] + sfs["1"][i] + sfs["2"][i] + sfs["3"][i] + 
                           sfs["4"][i] ) / 5
        sbs_averages[i] = (sbs["0"][i] + sbs["1"][i] + sbs["2"][i] + sbs["3"][i] + 
                           sbs["4"][i] ) / 5
    
    
    x_space = range(0,168)
    
    plt.plot(x_space,sfs["0"] , 'lightgreen') # plotting t, a separately 
    plt.plot(x_space,sfs["1"], 'lightgreen') # plotting t, b separately 
    plt.plot(x_space, sfs["2"], 'lightgreen') # plotting t, c separately 
    plt.plot(x_space, sfs["3"], 'lightgreen') # plotting t, c separately 
    plt.plot(x_space, sfs["4"], 'lightgreen') # plotting t, c separately 
    plt.plot(x_space, sfs_averages, 'green') # plotting t, c separately     
    
    
    print(sfs["0"])
    print(len(sfs["0"]))
    print(sbs["0"])
    print(len(sbs["0"]))
    
    plt.plot(x_space, sbs["0"] , 'lightcoral') # plotting t, a separately 
    plt.plot(x_space, sbs["1"], 'lightcoral') # plotting t, b separately 
    plt.plot(x_space, sbs["2"], 'lightcoral') # plotting t, c separately 
    plt.plot(x_space, sbs["3"], 'lightcoral') # plotting t, c separately 
    plt.plot(x_space, sbs["4"], 'lightcoral') # plotting t, c separately 
    plt.plot(x_space, sbs_averages, 'r') # plotting t, c separately
 
 
        
svm_sfs_accuracies = svm_sfs["all_accs"]
averages_sfs = [0] * 168
for i in range(0,168):
    averages_sfs[i] = (svm_sfs["all_accs"]["0"][i] + svm_sfs["all_accs"]["1"][i] + svm_sfs["all_accs"]["2"][i] + svm_sfs["all_accs"]["3"][i] + svm_sfs["all_accs"]["4"][i] ) / 5
    
    
# x_space = range(0,168)
# plt.plot(x_space,svm_sfs["all_accs"]["0"] , 'g--') # plotting t, a separately 
# plt.plot(x_space, svm_sfs["all_accs"]["1"], 'g--') # plotting t, b separately 
# plt.plot(x_space, svm_sfs["all_accs"]["2"], 'g--') # plotting t, c separately 
# plt.plot(x_space, svm_sfs["all_accs"]["3"], 'g--') # plotting t, c separately 
# plt.plot(x_space, svm_sfs["all_accs"]["4"], 'g--') # plotting t, c separately 
# plt.plot(x_space, averages_sfs, 'lime') # plotting t, c separately 

plot_graph(svm_sfs["all_accs"], svm_sbs["all_accs"])

plt.show()
