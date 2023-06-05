
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
    figure = plt.figure()
    sfs_averages = [0]*168
    sbs_averages = [0]*168
    
    for i in range(0,168):
        sfs_averages[i] = (sfs["0"][i] + sfs["1"][i] + sfs["2"][i] + sfs["3"][i] + 
                           sfs["4"][i] ) / 5
        sbs_averages[i] = (sbs["0"][i] + sbs["1"][i] + sbs["2"][i] + sbs["3"][i] + 
                           sbs["4"][i] ) / 5
    
    
    x_space = range(1,169)
    
    plt.plot(x_space,sfs["0"] , c='b', alpha =0.3) # plotting t, a separately 
    plt.plot(x_space,sfs["1"], c='b', alpha =0.3) # plotting t, b separately 
    plt.plot(x_space, sfs["2"], c='b', alpha =0.3) # plotting t, c separately 
    plt.plot(x_space, sfs["3"], c='b', alpha =0.3) # plotting t, c separately 
    plt.plot(x_space, sfs["4"], c='b', alpha =0.3) # plotting t, c separately 
    plt.plot(x_space, sfs_averages, 'b') # plotting t, c separately     
    
    
    # print(sfs["0"])
    # print(len(sfs["0"]))
    # print(sbs["0"])
    # print(len(sbs["0"]))
    
    plt.plot(x_space, sbs["0"] , c='r', alpha = 0.3) # plotting t, a separately 
    plt.plot(x_space, sbs["1"], c='r', alpha = 0.3) # plotting t, b separately 
    plt.plot(x_space, sbs["2"], c='r', alpha = 0.3) # plotting t, c separately 
    plt.plot(x_space, sbs["3"], c='r', alpha = 0.3) # plotting t, c separately 
    plt.plot(x_space, sbs["4"], c='r', alpha = 0.3) # plotting t, c separately 
    plt.plot(x_space, sbs_averages, 'r') # plotting t, c separately
    return figure
 
 
 # Find average number of features and average accuracy
# def find_averages(accuracies, features):
 
        
# svm_sfs_accuracies = svm_sfs["all_accs"]
# averages_sfs = [0] * 168
# for i in range(0,168):
#     averages_sfs[i] = (svm_sfs["all_accs"]["0"][i] + svm_sfs["all_accs"]["1"][i] + svm_sfs["all_accs"]["2"][i] + svm_sfs["all_accs"]["3"][i] + svm_sfs["all_accs"]["4"][i] ) / 5
    
    
# x_space = range(0,168)
# plt.plot(x_space,svm_sfs["all_accs"]["0"] , 'g--') # plotting t, a separately 
# plt.plot(x_space, svm_sfs["all_accs"]["1"], 'g--') # plotting t, b separately 
# plt.plot(x_space, svm_sfs["all_accs"]["2"], 'g--') # plotting t, c separately 
# plt.plot(x_space, svm_sfs["all_accs"]["3"], 'g--') # plotting t, c separately 
# plt.plot(x_space, svm_sfs["all_accs"]["4"], 'g--') # plotting t, c separately 
# plt.plot(x_space, averages_sfs, 'lime') # plotting t, c separately 

print("KNN SBS length =", len(knn_sbs["all_accs"]["0"]),"KNN SFS length =", len(knn_sfs["all_accs"]["0"] ))
print("KNN SBS length =", len(knn_sbs["all_accs"]["1"]),"KNN SFS length =", len(knn_sfs["all_accs"]["1"] ))
print("KNN SBS length =", len(knn_sbs["all_accs"]["2"]),"KNN SFS length =", len(knn_sfs["all_accs"]["2"] ))
print("KNN SBS length =", len(knn_sbs["all_accs"]["3"]),"KNN SFS length =", len(knn_sfs["all_accs"]["3"] ))
print("KNN SBS length =", len(knn_sbs["all_accs"]["4"]),"KNN SFS length =", len(knn_sfs["all_accs"]["4"] ))

knn_graph  = plot_graph(knn_sfs["all_accs"], knn_sbs["all_accs"])
knn_graph.suptitle('KNN feature selection performance')

print("NN SBS length =", len(nn_sbs["all_accs"]),"NN SFS length =", len(nn_sfs["all_accs"] ))

print("NN SBS length =", len(nn_sbs["all_accs"]["0"]),"NN SFS length =", len(nn_sfs["all_accs"]["0"] ))
print("NN SBS length =", len(nn_sbs["all_accs"]["1"]),"NN SFS length =", len(nn_sfs["all_accs"]["1"] ))
print("NN SBS length =", len(nn_sbs["all_accs"]["2"]),"NN SFS length =", len(nn_sfs["all_accs"]["2"] ))
print("NN SBS length =", len(nn_sbs["all_accs"]["3"]),"NN SFS length =", len(nn_sfs["all_accs"]["3"] ))
print("NN SBS length =", len(nn_sbs["all_accs"]["4"]),"NN SFS length =", len(nn_sfs["all_accs"]["4"] ))
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
nn_graph  = plot_graph(nn_sfs["all_accs"], nn_sbs["all_accs"])
nn_graph.suptitle('NN feature selection performance')

print("RF SBS length =", len(rf_sbs["all_accs"]),"RF SFS length =", len(rf_sfs["all_accs"] ))

plt.xlabel("Number of features")
plt.ylabel("Accuracy")

rf_graph  = plot_graph(rf_sfs["all_accs"], rf_sbs["all_accs"])
rf_graph.suptitle('RF feature selection performance')
print("SVM SBS length =", len(svm_sbs["all_accs"]),"SVM SFS length =", len(svm_sfs["all_accs"] ))

plt.xlabel("Number of features")
plt.ylabel("Accuracy")

svm_graph  = plot_graph(svm_sfs["all_accs"], svm_sbs["all_accs"])
svm_graph.suptitle('SVM feature selection performance')

plt.xlabel("Number of features")
plt.ylabel("Accuracy")
knn_graph.show()
nn_graph.show()
rf_graph.show()
svm_graph.show()
input()
