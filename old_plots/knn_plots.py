import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import plots

plt.close("all")

###### KNN SFS #######
file_knn_SFS = open('knn_SFS.json', 'r')
SFS = json.load(file_knn_SFS)

# Get features frequencies
knn_SFS_feature_freqs = SFS["knn_freqs"]
# Get accruacies / #features
knn_SFS_accuracy = SFS["all_accs"]

sfs_accuracy = SFS['all_accs']
sfs_frequencies = SFS["knn_freqs"]

# Read file
file_SBS = open('knn_SBS.json', 'r')
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



plots.bar_plot(sfs_freq,x_data, True)
plots.bar_plot(sbs_freq, x_data, False)
plots.graph_plot(l, sfs_accuracy, x_data)

plots.create_table('knn_sfs_table.csv', sfs_frequencies)
plots.create_table('knn_sbs_table.csv', sbs_frequencies)

# fig = plt.figure()
# ax = fig.add_subplot(111)

# x=[1,2,3,4,5,6,7,8,9,10]
# y=[1,1,1,2,10,2,1,1,1,1]
# line, = ax.plot(x, y)

# ymax = max(y)
# xpos = y.index(ymax)
# xmax = x[xpos]

# ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+5),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )

# ax.set_ylim(0,20)
# plt.show()





# ## Accuracy graph
# # Features list
# x_data = [int(key) for key in knn_SFS_accuracy.keys()]

# # Accuracy lists

# figure_acc = plt.figure()
# ax_acc = figure_acc.add_subplot(111)
# sfs_highest_acc = max(knn_SFS_all_accs_data)
# feature_number_at_max_index = knn_SFS_all_accs_data.index(sfs_highest_acc)
# feature_number_at_max = x_data[feature_number_at_max_index]

# # Add max annotaiton
# ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+5),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )


# # Create a line plot using Matplotlib
# plt.plot(x_data, knn_SFS_all_accs_data, 'r--',x_data, knn_SBS_all_accs_data, 'b--')
# plt.xlabel('Number of features')
# plt.ylabel('Accuracy')
# plt.title('Performance Forward Selection')
# plt.show()












# ###################### KNN SBS  data extraction #####################333
# # file_knn_SBS = open('knn_SBS.json', 'r')
# # knn_SBS = json.load(file_knn_SBS)

# # # Get feature frequencies
# # knn_SBS_feature_freqs = knn_SFS["knn_freqs"]
# # # Get accruacies / #features
# # knn_SBS_accuracy = knn_SFS["all_accs"]


# ##################3 Bar graph ##########################
# # Feature frequency lists
# knn_SFS_feature_freqs_list = [0] * 170

# for key in knn_SFS_feature_freqs:
#     knn_SFS_feature_freqs_list[key] = knn_SFS_feature_freqs[key]

# knn_SFS_feature_freqs_list = [random.random()] *170 # Placeholder

# ##### Actual bar plotting
# fig_sfs_bar = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# students = [23,17,35,29,12]
# # Create list of string of every feature for label
# label_features = list(i for i in range(0,170))
# label_features= map(str, label_features)

# ax.bar(label_features,knn_SFS_feature_freqs_list)
# plt.show()




# # {
# #     "knn_freqs": {"151": 2, "162": 2, "130": 1, "138": 1, "145": 1, "158": 1, "160": 1, "161": 1,
# #                "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0,
# #                "10": 0, "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0,
# #                "19": 0, "20": 0, "21": 0, "22": 0, "23": 0, "24": 0, "25": 0, "26": 0, "27": 0, 
# #                "28": 0, "29": 0, "30": 0, "31": 0, "32": 0, "33": 0, "34": 0, "35": 0, "36": 0,
# #                "37": 0, "38": 0, "39": 0, "40": 0, "41": 0, "42": 0, "43": 0, "44": 0, "45": 0,
# #                "46": 0, "47": 0, "48": 0, "49": 0, "50": 0, "51": 0, "52": 0, "53": 0, "54": 0, 
# #                "55": 0, "56": 0, "57": 0, "58": 0, "59": 0, "60": 0, "61": 0, "62": 0, "63": 0,
# #                "64": 0, "65": 0, "66": 0, "67": 0, "68": 0, "69": 0, "70": 0, "71": 0, "72": 0,
# #                "73": 0, "74": 0, "75": 0, "76": 0, "77": 0, "78": 0, "79": 0, "80": 0, "81": 0, 
# #                "82": 0, "83": 0, "84": 0, "85": 0, "86": 0, "87": 0, "88": 0, "89": 0, "90": 0, 
# #                "91": 0, "92": 0, "93": 0, "94": 0, "95": 0, "96": 0, "97": 0, "98": 0, "99": 0, 
# #                "100": 0, "101": 0, "102": 0, "103": 0, "104": 0, "105": 0, "106": 0, "107": 0, 
# #                "108": 0, "109": 0, "110": 0, "111": 0, "112": 0, "113": 0, "114": 0, "115": 0, 
# #                "116": 0, "117": 0, "118": 0, "119": 0, "120": 0, "121": 0, "122": 0, "123": 0, 
# #                "124": 0, "125": 0, "126": 0, "127": 0, "128": 0, "129": 0, "131": 0, "132": 0, 
# #                "133": 0, "134": 0, "135": 0, "136": 0, "137": 0, "139": 0, "140": 0, "141": 0, 
# #                "142": 0, "143": 0, "144": 0, "146": 0, "147": 0, "148": 0, "149": 0, "150": 0, 
# #                "152": 0, "153": 0, "154": 0, "155": 0, "156": 0, "157": 0, "159": 0, "163": 0, 
# #                "164": 0, "165": 0, "166": 0, "167": 0, "168": 0, "169": 0}, 
 
# #     "knn_accuracy": 0.7440268065268065, 
# #     "knn_old_accuracy": 0.6768, 
# #     "knn_num_features": {"1": 5, "2": 5}, 
# #     "knn_cum_time": 12.967894999999999, 
# #     "all_accs": {"0": 0.6712059869954607, "1": 0.7055959391485707, "2": 0.7224381977671451, "3": 0.7368191019506808, "4": 0.7440268065268065, 
# #                  "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0, "10": 0.0, "11": 0.0, "12": 0.0, "13": 0.0, "14": 0.0, "15": 0.0, "16": 0.0, 
# #                  "17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "22": 0.0, "23": 0.0, "24": 0.0, "25": 0.0, "26": 0.0, "27": 0.0, "28": 0.0, 
# #                  "29": 0.0, "30": 0.0, "31": 0.0, "32": 0.0, "33": 0.0, "34": 0.0, "35": 0.0, "36": 0.0, "37": 0.0, "38": 0.0, "39": 0.0, "40": 0.0, 
# #                  "41": 0.0, "42": 0.0, "43": 0.0, "44": 0.0, "45": 0.0, "46": 0.0, "47": 0.0, "48": 0.0, "49": 0.0, "50": 0.0, "51": 0.0, "52": 0.0, 
# #                  "53": 0.0, "54": 0.0, "55": 0.0, "56": 0.0, "57": 0.0, "58": 0.0, "59": 0.0, "60": 0.0, "61": 0.0, "62": 0.0, "63": 0.0, "64": 0.0, 
# #                  "65": 0.0, "66": 0.0, "67": 0.0, "68": 0.0, "69": 0.0, "70": 0.0, "71": 0.0, "72": 0.0, "73": 0.0, "74": 0.0, "75": 0.0, "76": 0.0, 
# #                  "77": 0.0, "78": 0.0, "79": 0.0, "80": 0.0, "81": 0.0, "82": 0.0, "83": 0.0, "84": 0.0, "85": 0.0, "86": 0.0, "87": 0.0, "88": 0.0, 
# #                  "89": 0.0, "90": 0.0, "91": 0.0, "92": 0.0, "93": 0.0, "94": 0.0, "95": 0.0, "96": 0.0, "97": 0.0, "98": 0.0, "99": 0.0, "100": 0.0, 
# #                  "101": 0.0, "102": 0.0, "103": 0.0, "104": 0.0, "105": 0.0, "106": 0.0, "107": 0.0, "108": 0.0, "109": 0.0, "110": 0.0, "111": 0.0, 
# #                  "112": 0.0, "113": 0.0, "114": 0.0, "115": 0.0, "116": 0.0, "117": 0.0, "118": 0.0, "119": 0.0, "120": 0.0, "121": 0.0, "122": 0.0, 
# #                  "123": 0.0, "124": 0.0, "125": 0.0, "126": 0.0, "127": 0.0, "128": 0.0, "129": 0.0, "130": 0.0, "131": 0.0, "132": 0.0, "133": 0.0, 
# #                  "134": 0.0, "135": 0.0, "136": 0.0, "137": 0.0, "138": 0.0, "139": 0.0, "140": 0.0, "141": 0.0, "142": 0.0, "143": 0.0, "144": 0.0, 
# #                  "145": 0.0, "146": 0.0, "147": 0.0, "148": 0.0, "149": 0.0, "150": 0.0, "151": 0.0, "152": 0.0, "153": 0.0, "154": 0.0, "155": 0.0, 
# #                  "156": 0.0, "157": 0.0, "158": 0.0, "159": 0.0, "160": 0.0, "161": 0.0, "162": 0.0, "163": 0.0, "164": 0.0, "165": 0.0, "166": 0.0, 
# #                  "167": 0.0, "168": 0.0, "169": 0.0}
# #     }