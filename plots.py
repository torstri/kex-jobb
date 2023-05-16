import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from jinja2 import *
import random
from IPython.display import display

plt.close("all")


def bar_plot(features, x_data, sfs = True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(x_data, features)
    if(sfs):
        plt.title('Appearances for every featur: SFS')
    else:
        plt.title('Appearances for every feature: SBS')
    plt.xlabel('Feature number')
    plt.ylabel('Appearances')
    plt.plot()
    plt.show()
    

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
    
    
def create_table(filename, y_data):
    
    df = pd.DataFrame.from_dict(y_data, orient='index')
    print(df)
    df.reset_index()
    
    print(df)
    labels = list(y_data.keys())
    print("Features: ",labels)
    values = list(y_data.values())
    print("Värden: ",values)
    
    dict = {"Feature": labels, "Occurence(s)": values}

    # df = pd.DataFrame.from_dict(dict, orient='index')
    df = pd.DataFrame(y_data.items(), columns=['Feature', 'Occurence'])
    print("DF =",df)
    
    df.to_csv(filename, index=False, header=True)
    
def create_big_bar():

    file_names = ['nn_SFS.json', 'nn_SBS.json', 
                'knn_SFS.json', 'knn_SBS.json', 
                'svm_SFS.json', 'svm_SBS.json', 
                'rf_SFS.json', 'rf_SBS.json']

    key_names = ['nn_freqs', 'knn_freqs', 'svm_freqs','rf_freqs']

    all_freqs ={}
    for i in range(0,170):
        all_freqs[str(i)] = 0
    j = 0

    # print("all_freqs)
    for i in range(0,8):
        
        file_name = file_names[i]
        f = open(file_name, 'r')
        SFS = json.load(f)
        key_name = key_names[j]
        if(i % 2 == 1):
            j+= 1
        frequencies = SFS[key_name]
        
        print("Freqs = ", frequencies)
        for key in frequencies:
            all_freqs[key] += frequencies.get(key)


    fr = [0] * 170
    x = [i for i in range(0,170)]
    for key in all_freqs:
        fr[int(key)] = all_freqs.get(key)


    print("x = ",x)

    print("asdf =", all_freqs)

    print("fr = ", fr)
    bar_plot(fr,x)    

    sorted_freqs = dict(sorted(all_freqs.items(), key=lambda x:x[1], reverse=True))

    print("Sorted_freqs: ",sorted_freqs)

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