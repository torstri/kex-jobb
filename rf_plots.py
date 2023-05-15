import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random

plt.close("all")

###### KNN SFS #######
file_SFS = open('rf_SFS.json', 'r')
SFS = json.load(file_SFS)

# Get features frequencies
SFS_feature_freqs = SFS["rf_freqs"]
# Get accruacies / #features
SFS_accuracy = SFS["all_accs"]

sfs_accuracy = SFS['all_accs']
sfs_frequencies = SFS["rf_freqs"]

# Read file
file_SBS = open('rf_SBS.json', 'r')
SBS = json.load(file_SBS)

# sbs_accuracy = SBS['all_accs'] # Get accuracies
# sbs_accuracy = list(sbs_accuracy.values())
# sbs_frequencies = SBS["knn_freqs"] # Get frequencies
sbs_freq = []

for i in range(0,170):
    sbs_freq.append(random.randint(0,5))

print("SBS_freqs =",sbs_freq)

sfs_accuracy = list(sfs_accuracy.values())

l = [] # Placeholder for sbs

x_data = list(i for i in range(0,170)) # Create list containing 0, 1, ... , 169
sfs_freq = [0] * 170
for key in sfs_frequencies:
    sfs_freq[int(key)] = sfs_frequencies.get(key)

i = 0
for value in sfs_freq:
    print(i,": Value = ", value)
    i += 1
    

for i in range(0,170):
    l.append(random.uniform(0.60,0.76))
    print(i,": SBS = ", l[i], " SFS = ",sfs_accuracy[i] )


def bar_plot_sfs(features, x_data, sfs = True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_data, features)
    if(sfs):
        plt.title('Appearances for every featur: SFS')
    else:
        plt.title('Appearances for every featur: SBS')
    plt.xlabel('Feature number')
    plt.ylabel('Appearances')
    plt.plot()
    

def graph_plot(sbs, sfs, x_data):
    fig = plt.figure() # Create plot
    ax = fig.add_subplot(111)
    sfs_max = max(sfs)
    sbs_max = max(sbs)
    sfs_x = sfs.index(sfs_max)
    sbs_x = sbs.index(sbs_max)
    # Add max annotaiton
    ax.annotate('SFS local max', xy=(sfs_x, sfs_max), xytext=(sfs_x, sfs_max + 0.1),
                arrowprops=dict(facecolor='blue', shrink=0.005),
                )
    ax.annotate('SBS local max', xy=(sbs_x, sbs_max), xytext=(sbs_x, sbs_max + 0.1),
                arrowprops=dict(facecolor='red', shrink=0.005),
                )
    # Create a line plot using Matplotlib
    plt.plot(x_data, sfs, 'r--',x_data, sbs, 'b--')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('Average Performance K-nearest neighbours')
    ax.set_ylim(0.5,1)
    plt.show()

bar_plot_sfs(sfs_freq,x_data, True)
bar_plot_sfs(sbs_freq, x_data, False)
graph_plot(l, sfs_accuracy, x_data)
