import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import plots

plt.close("all")

###### KNN SFS #######
file_SFS = open('svm_SFS.json', 'r')
SFS = json.load(file_SFS)



sfs_accuracy = SFS['all_accs']
sfs_frequencies = SFS["svm_freqs"]

# Read file
file_SBS = open('svm_SBS.json', 'r')
SBS = json.load(file_SBS)

# sbs_accuracy = SBS['all_accs'] # Get accuracies
# sbs_accuracy = list(sbs_accuracy.values())
# sbs_frequencies = SBS["knn_freqs"] # Get frequencies
sbs_freq = []

for i in range(0,170):
    sbs_freq.append(random.randint(0,5))

# print("SBS_freqs =",sbs_freq)

sfs_accuracy = list(sfs_accuracy.values())

l = [] # Placeholder for sbs

x_data = list(i for i in range(0,170)) # Create list containing 0, 1, ... , 169
sfs_freq = [0] * 170
for key in sfs_frequencies:
    sfs_freq[int(key)] = sfs_frequencies.get(key)

i = 0
for value in sfs_freq:
    # print(i,": Value = ", value)
    i += 1
    

for i in range(0,170):
    l.append(random.uniform(0.60,0.76))
    # print(i,": SBS = ", l[i], " SFS = ",sfs_accuracy[i] )




# plots.bar_plot(sfs_freq,x_data, True)
# plots.bar_plot(sbs_freq, x_data, False)
# plots.graph_plot(l, sfs_accuracy, x_data)
# print("Frequencies = ", sfs_frequencies)
# probs = []

# df_probs = pd.DataFrame(columns=['Feature', 'Occurence(s)', 'Probability'])

# for index, feature in enumerate(sfs_frequencies):
    
#     occurence = sfs_frequencies[feature]
#     probability = float(occurence)/5.0
    
#     print("Index =", index, " Feature =", feature, "Occurence =", occurence)
#     df_probs.loc[index] = [feature, occurence, probability]
    
# print(df_probs)   
# df_probs.to_csv('svm_sfs_probabilities.csv') 


# sfs_frequencies["Probabilities"] = probs

# plots.create_table('svm_sfs_table.csv', sfs_frequencies, 'Occurences', 'Feature')


# plots.create_table('svm_sfs_table.csv', sfs_frequencies)
# plots.create_table('svm_sbs_table.csv', sbs_frequencies)
