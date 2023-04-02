import os
import cv2
from border_detection import border_detection
import dullrazor
# import dullrazor
from skimage import measure
from melanoma_classifier.src import _get_tabular_dataframe_utils as tab
from sklearn.datasets import make_classification
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from timeit import default_timer as timer
import pandas as pd
import tensorflow as tf
import get_config as conf
import matplotlib.pyplot as plt

def print_statistics(y_test, y_pred):
  print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
  print("Balanced accurancy: ", metrics.balanced_accuracy_score(y_test, y_pred))
  print("Precision: ", metrics.precision_score(y_test, y_pred))
  print("Recall: ", metrics.recall_score(y_test, y_pred))
  print("F1: ", metrics.f1_score(y_test, y_pred))

def read_csv(path):
  x = np.zeros(10015)
  i = 0
  with open(path, 'r') as file:
    next(file)
    csvreader = csv.reader(file)
    for row in csvreader:
      if(row[1] == '1.0'):
        x[i] = 1
      else:
        x[i] = 0
      i += 1
  return x

# def get_ratio(y):
#   y_sum = sum(y)
#   return (len(y) - y_sum) / y_sum

def construct_df(test_size):
  data = []
  for i in range(24306, 24306+test_size):
    # result = "malignant" if y[i-24306] == 1 else "benign"
    data.append(["/ISIC_00" + str(i) + ".jpg"]) # , result
  df = pd.DataFrame(data, columns=["filename"]) # , "target"
  return df

def create_train_and_test_SVM(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=79) # 70% training and 30% test
  clf = svm.SVC(kernel='linear', class_weight={1: 7}) # Linear Kernel
  clf.fit(x_train, y_train)
  curr_time = timer()
  print("Time after fitting: ", curr_time - start)
  y_pred = clf.predict(x_test)

  print(y_pred)
  print_statistics(y_test, y_pred)

start = timer()

test_size = 1000  # efter 34180 är det whack

df = construct_df(test_size)
cfg = conf.get_config()

features = tab.get_tabular_features(cfg, df, "./dataset/archive-2/images", "./dataset/archive-2/segmentations")

x = features.drop(columns=["filename"]).to_numpy()  # we do not modify features
y = read_csv("./dataset/archive-2/GroundTruth.csv")[:test_size]

print(x.shape)

for j in range(0, x.shape[1]):
  plt.title(list(features.columns)[j+1])
  for i in range(0, test_size):
    color = "r+" if y[i] == 1 else "bo"
    plt.plot(x[i][j], i, color)
  plt.show()

create_train_and_test_SVM(x, y)

end = timer()
print("\nTime for execution: ", end - start)