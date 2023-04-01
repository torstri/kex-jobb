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



# def create_array(path):
#   test_size = 5
#   x = np.zeros((test_size, 13)) # number of images and number of features
#   y = read_csv(path + "/GroundTruth.csv")
#   y = y[:test_size]
  
#   ratio = get_ratio(y)

#   imgs_path = path + "/images"
#   seg_imgs_path = path + "/masks"

#   for img_number in range(24306, 24306+test_size):  # 34320 is the maximum
#     img_path = imgs_path + "/ISIC_00" + str(img_number) + ".jpg"
#     seg_img_path = seg_imgs_path + "/ISIC_00" + str(img_number) + "_segmentation.png"
#     features = feature_extraction(img_path, seg_img_path) # Bild 26042 är whack eftersom hela bilden är utslaget
#     i = 0
#     for feature_list in features:
#       for feature in feature_list:
#         x[img_number-24306][i] = feature
#         i += 1

#   curr_time = timer()
#   print("Time for feature extraction: ", curr_time - start)

#   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109) # 70% training and 30% test

#   clf = svm.SVC(kernel='linear', class_weight={1: ratio}) # Linear Kernel

#   clf.fit(x_train, y_train)

#   curr_time = timer()
#   print("Time after fitting: ", curr_time - start)

#   y_pred = clf.predict(x_test)

#   print(y_pred)

#   print_statistics(y_test, y_pred)

#def train_and_test_SVM(features):


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
  y = read_csv("./dataset/archive-2/GroundTruth.csv")
  for i in range(24306, 24306+test_size):

    # result = "malignant" if y[i-24306] == 1 else "benign"
    # data.append(["/ISIC_00" + str(i) + ".jpg", result])
    # If we want result in df as well.
    data.append("/ISIC_00" + str(i) + ".jpg")
  df = pd.DataFrame(data, columns=["filename"]) # , "target" if result is wanted
  return df

start = timer()

df = construct_df(100)
cfg = conf.get_config()

features = tab.get_tabular_features(cfg, df, "./dataset/archive-2/images", "./dataset/archive-2/masks")

print(features)

#train_and_test_SVM(features)

end = timer()
print("\nTime for execution: ", end - start)