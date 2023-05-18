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
    
    
# Första RQ
def create_probability_tables():
    
    # Files to store result
    target_file_names = ['nn_total_probability.csv',
                         'knn_total_probability.csv',
                         'svm_total_probability.csv',
                         'rf_total_probability.csv']
    
    # Files to retrieve data from
    file_names = ['nn_SFS.json', 'nn_SBS.json', 
                'knn_SFS.json', 'knn_SBS.json', 
                'svm_SFS.json', 'svm_SBS.json', 
                'rf_SFS.json', 'rf_SBS.json']

    # Keys in dicts
    key_names = ['nn_freqs', 'knn_freqs', 'svm_freqs','rf_freqs']
    
    
    j = 0 # key index
    i = 0
    target_number = 0
    while i < 8:
        
        file_name = file_names[i]
        file_name2 = file_names[i+1]
        f = open(file_name, 'r')
        f2 = open(file_name2, 'r')
        SFS = json.load(f)
        SBS = json.load(f2)
        key_name = key_names[j]
            
        frequencies_sfs = SFS[key_name]
        frequencies_sbs = SBS[key_name]
        df_feature_probs = pd.DataFrame(columns=['Feature', 'Occurence', 'Probability', 'Probability in SFS', 'Probability in SBS'])

        for index, feature in enumerate(frequencies_sfs):
            
                occurence_sfs = frequencies_sfs[feature]
                occurence_sbs = frequencies_sbs[feature]
                occurence_total = occurence_sbs + occurence_sfs

                probability_sfs = float(occurence_sfs)/5.0
                probability_sbs = float(occurence_sbs)/5.0
                probability_total = float(occurence_total)/10.0
            
                df_feature_probs.loc[index] = [feature, occurence_total, probability_total, probability_sfs, probability_sbs]

        df_feature_probs = df_feature_probs.sort_values(by=['Occurence'], ascending = False)
        df_feature_probs.to_csv(target_file_names[target_number])
        target_number += 1     
        j += 1
        i += 2
    
    
# Andra RQ
# # def create_universal_table():
# #     print("")
    
# #     file_names = ['nn_SFS.json', 'nn_SBS.json', 
# #                 'knn_SFS.json', 'knn_SBS.json', 
# #                 'svm_SFS.json', 'svm_SBS.json', 
# #                 'rf_SFS.json', 'rf_SBS.json']

# #     key_names = ['nn_freqs', 'knn_freqs', 'svm_freqs','rf_freqs']
# #     all_freqs ={}
    
# #     j = 0

# #     for i in range(0,8):
        
# #         file_name = file_names[i]
# #         f = open(file_name, 'r')
# #         SFS = json.load(f)
# #         key_name = key_names[j]
# #         if(i % 2 == 1):
# #             j+= 1
# #         frequencies = SFS[key_name]
        

# #         for index, feature in enumerate(frequencies):
            
# #                 occurence = frequencies[feature]
# #                 probability = float(occurence)/5.0
            

# #                 all_freqs[feature] += frequencies.get(feature)

# #         df_feature_probs.to_csv(target_file_names[i]) 

# #     fr = [0] * 170
# #     x = [i for i in range(0,170)]
# #     for key in all_freqs:
# #         fr[int(key)] = all_freqs.get(key)

    
# #     all_freqs = dict(sorted(all_freqs.items(), key=lambda x:x[1], reverse=True))
# #     df_all_probs = pd.DataFrame(columns=['Feature', 'Occurence(s)', 'Probability'])
# #     abcd_features = 0
# #     sift_features = 0
    
# #     for index, feature in enumerate(all_freqs):
# #         occurence = all_freqs[feature]
# #         probability = float(occurence)/40.0
# #         df_all_probs.loc[index] = [feature, occurence, probability]
# #         if( int(feature) > 149):
# #             abcd_features += all_freqs.get(feature)
# #         else:
# #             sift_features +=  all_freqs.get(feature)    
    
# #     bar_plot(fr, x, 'Occurences for every feature', 'Features', 'Occurences') 

# #     df_all_probs.to_csv('big_data_new.csv')

    
# #     create_table('big_data.csv', sorted_freqs)
# #     feature_diff = {'ABCD':abcd_features, 'SIFT':sift_features}
# #     df_diff = pd.DataFrame(feature_diff.items(), columns=['Feature Type', 'Occurence(s)'])
# #     df_diff['Representation'] = [float(abcd_features)/float((169-150)), float(sift_features)/float(150)]
# #     df_diff.to_csv('representation.csv', index=False, header=True)
    

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
    
    
    
    
        
        
        
    
create_probability_tables()
# generate_tables()
# plot_all_graphs()

# create_big_bar()
