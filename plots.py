import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from jinja2 import *
import random
from IPython.display import display

plt.close("all")


def bar_plot(y_data,x_data, title, labelx, labely):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_data, y_data)
    plt.title(title)

    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.plot()
    plt.show()
    

def graph_plot(sbs, sfs, x_data, label):
    fig = plt.figure() # Create plot
    ax = fig.add_subplot(111)
    sfs_max = max(sfs)
    sbs_max = max(sbs)
    sfs_x = sfs.index(sfs_max)
    sbs_x = sbs.index(sbs_max)

    ax.annotate('SFS local max',
             xy=(sfs_x, sfs_max),
             xytext=(sfs_x, sfs_max + 0.05),
             arrowprops=dict(arrowstyle='->',shrinkA = 9, shrinkB = 8,
                             facecolor='red',
                             edgecolor='red',
                             linewidth=2.5,
                             mutation_scale=20),
             )
    

                
                
    ax.annotate('SBS local max',
             xy=(sbs_x, sbs_max),
             xytext=(sbs_x, sbs_max - 0.1),
             arrowprops=dict(arrowstyle='->',
                             facecolor='blue',
                             edgecolor='blue',
                             linewidth=2.5,
                             mutation_scale=20),
             )
    # Create a line plot using Matplotlib
    plt.plot(x_data, sfs, 'r--',x_data, sbs, 'b--')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title(label)
    ax.set_ylim(0.55,0.9)
    ax.set_xlim(0,170)
    i = 0
    x_points = [0]
    while i < 17:
        
        i += 1
        x_points.append(x_points[i - 1] + 10)
    # ax.set_xscale(10, 'linear')
    plt.xticks(x_points)
    plt.axvline(sfs_x, ls=':', c='k')
    plt.axvline(sbs_x, ls=':', c='k')
    
    plt.show()
    
    

    
def create_big_bar():

    file_names = ['nn_SFS.json', 'nn_SBS.json', 
                'knn_SFS.json', 'knn_SBS.json', 
                'svm_SFS.json', 'svm_SBS.json', 
                'rf_SFS.json', 'rf_SBS.json']

    key_names = ['nn_freqs', 'knn_freqs', 'svm_freqs','rf_freqs']
    
    target_file_names = ['nn_sfs_probability.csv', 'nn_sbs_probability.csv',
                         'knn_sfs_probability.csv', 'knn_sbs_probability.csv',
                         'svm_sfs_probability.csv', 'svm_sbs_probability.csv',
                         'rf_sfs_probability.csv', 'rf_sbs_probability.csv']

    all_freqs ={}
    for i in range(0,170):
        all_freqs[str(i)] = 0
    j = 0

    for i in range(0,8):
        
        file_name = file_names[i]
        f = open(file_name, 'r')
        SFS = json.load(f)
        key_name = key_names[j]
        if(i % 2 == 1):
            j+= 1
        frequencies = SFS[key_name]
        
        df_feature_probs = pd.DataFrame(columns=['Feature', 'Occurence(s)', 'Probability'])

        for index, feature in enumerate(frequencies):
            
                occurence = frequencies[feature]
                probability = float(occurence)/5.0
            
                df_feature_probs.loc[index] = [feature, occurence, probability]

                all_freqs[feature] += frequencies.get(feature)

        df_feature_probs.to_csv(target_file_names[i]) 

    fr = [0] * 170
    x = [i for i in range(0,170)]
    for key in all_freqs:
        fr[int(key)] = all_freqs.get(key)

    
    all_freqs = dict(sorted(all_freqs.items(), key=lambda x:x[1], reverse=True))
    df_all_probs = pd.DataFrame(columns=['Feature', 'Occurence(s)', 'Probability'])
    abcd_features = 0
    sift_features = 0
    
    for index, feature in enumerate(all_freqs):
        occurence = all_freqs[feature]
        probability = float(occurence)/40.0
        df_all_probs.loc[index] = [feature, occurence, probability]
        if( int(feature) > 149):
            abcd_features += all_freqs.get(feature)
        else:
            sift_features +=  all_freqs.get(feature)    
    
    bar_plot(fr, x, 'Occurences for every feature', 'Features', 'Occurences') 

    df_all_probs.to_csv('big_data_new.csv')

    
    # create_table('big_data.csv', sorted_freqs)
    feature_diff = {'ABCD':abcd_features, 'SIFT':sift_features}
    df_diff = pd.DataFrame(feature_diff.items(), columns=['Feature Type', 'Occurence(s)'])
    df_diff['Representation'] = [float(abcd_features)/float((169-150)), float(sift_features)/float(150)]
    df_diff.to_csv('representation.csv', index=False, header=True)
    
    
    
    
def plot_all_graphs():
    x = [i for i in range(0,168)]
    
    file_names = ['nn_SFS.json', 'nn_SBS.json', 
            'knn_SFS.json', 'knn_SBS.json', 
            'svm_SFS.json', 'svm_SBS.json', 
            'rf_SFS.json', 'rf_SBS.json']

    
    i = 0
    while i < 8:
        label = ""
        if( i < 2):
            label = "Average Performance Neural Network"
        elif(i < 4):
            label = "Average Performance K-Nearest-Neighbours"
        elif(i < 6):
            label = "Average Performance Support Vector Machine"
        else:
            label = "Average Performance Random Forest"
        file_name = file_names[i]
        f = open(file_name, 'r')
        SFS = json.load(f)
        sfs_accuracies = SFS["all_accs"]
        
        file_name = file_names[i + 1]
        f = open(file_name, 'r')
        SBS = json.load(f)
        sbs_accuracies = SBS["all_accs"]
        
        sbs_accuracies.pop('168')
        sfs_accuracies.pop('168')
        sbs_accuracies.pop('169')
        sfs_accuracies.pop('169')
        
        
        i += 2
        sfs_list = [0] *168
        sbs_list = [0] *168
        
        for key in sfs_accuracies:
            sfs_list[int(key)] = sfs_accuracies.get(key)
            sbs_list[int(key)] = sbs_accuracies.get(key)
        graph_plot(sbs_list, sfs_list,x, label)

def create_table(filename, y_data, col1, col2):
    
    df = pd.DataFrame.from_dict(y_data, orient='index')
    
    

    # df = pd.DataFrame.from_dict(dict, orient='index')
    df = pd.DataFrame(y_data.items(), columns=[col1, col2])
    
    df.to_csv(filename, index=False, header=True)     
        
def generate_tables():
    file_names = ['nn_SFS.json', 'nn_SBS.json', 
        'knn_SFS.json', 'knn_SBS.json', 
        'svm_SFS.json', 'svm_SBS.json', 
        'rf_SFS.json', 'rf_SBS.json']
    
    key_names= ['nn_num_features','knn_num_features', 'svm_num_features', 'rf_num_features']
    
    target_files = ['nn_SFS_num.csv', 'nn_SBS_num.csv', 
        'knn_SFS_num.csv', 'knn_SBS_num.csv', 
        'svm_SFS_num.csv', 'svm_SBS_num.csv', 
        'rf_SFS_num.csv', 'rf_SBS_num.csv']
    
    col1 = [""]
    
    j = 0
    averages = {}
    for i in  range(0,8):
        file_name = file_names[i]
        key_name = key_names[j]
        target_file = target_files[i]
        
        if(i % 2 == 1):
            j += 1
        f = open(file_name, 'r')
        features = json.load(f)
        num_features = features[key_name]
        
        create_table(target_file, num_features, 'Test', 'Number of features')
        empty_list = [0] * 5
        for key in num_features:
            empty_list[int(key) - 1] = int(num_features.get(key))

        if(i == 0):
            averages["nn_SFS"] = sum(empty_list) / len(empty_list)
        elif(i == 1):
            averages["nn_SBS"] = sum(empty_list) / len(empty_list)
        elif(i == 2):
            averages["knn_SFS"] = sum(empty_list) / len(empty_list)
        elif(i == 3):
            averages["knn_SBS"] = sum(empty_list) / len(empty_list)
        elif(i == 4):
            averages["svm_SFS"] = sum(empty_list) / len(empty_list)
        elif(i == 5):
            averages["svm_SBS"] = sum(empty_list) / len(empty_list)
        elif(i == 6):
            averages["rf_SFS"] = sum(empty_list) / len(empty_list)
        else:
            averages["rf_SBS"] = sum(empty_list) / len(empty_list)
        
    df = pd.DataFrame(averages.items(), columns=['Selection Method', 'Average Number of Features'])
    df.to_csv('average_features_methods.csv', index=False, header=True)
    
    
    
    
        
        
        
    

# generate_tables()
# plot_all_graphs()

create_big_bar()

# for key in sfs_frequencies:
#     all_freqs[int(key)] += sfs_frequencies.get(key)

# for key in sfs_frequencies:
#     # print("Feature = ", key, "Value = ", all_freqs[int(key)], " Org Value =", sfs_frequencies.get(key))
#     if( all_freqs[int(key)] != sfs_frequencies.get(key)):
#         print("EEEER")
    

# for i in range(0,170):
# print("Hello")
# df = pd.DataFrame({
#     "strings": ["Adam", "Mike"],
#     "ints": [1, 3],
#     "floats": [1.123, 1000.23]
# })
# df.style \
#   .format(precision=3, thousands=".", decimal=",") \
#   .format_index(str.upper, axis=1)\
#   .relabel_index(["row 1", "row 2"], axis=0)    

# print("Hello")

# dict = {'Name' : ['Martha', 'Tim', 'Rob', 'Georgia'],
#         'Maths' : [87, 91, 97, 95],
#         'Science' : [83, 99, 84, 76]}
# df = pd.DataFrame(dict)

# display(df)


# df.head()