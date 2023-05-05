import os
import cv2
from border_detection import border_detection
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

# def get_ratio(y):
#   y_sum = sum(y)
#   return (len(y) - y_sum) / y_sum

def construct_df(test_size):
  data = []
  img_to_use = np.load('./SIFT/array_data/used.npy')
  for i in img_to_use:
    # result = "malignant" if y[i-24306] == 1 else "benign"
    data.append([str(i) + ".jpg"]) # , result
  df = pd.DataFrame(data, columns=["filename"]) # , "target"
  return df

start = timer()

test_size = 3297  # efter 34178 är det whack på HAM10000

df = construct_df(test_size)
cfg = conf.get_config()

features = tab.get_tabular_features(cfg, df, "./dataset/balanced/images/", "./dataset/balanced/segmentations/")

x = features.drop(columns=["filename"]).to_numpy()  # we do not modify features
np.save("./dataset/balanced/features.npy", x)
print(x.shape)

used_imgs = features["filename"].map(lambda x: int(''.join(x.split())[:-4])).to_numpy()
np.save("./dataset/balanced/used_imgs.npy", used_imgs)

print(used_imgs.shape)

curr_time = timer()
print("Time after feature extraction: ", curr_time-start)

y = np.load("./dataset/balanced/GroundTruth.npy")

# for j in range(0, x.shape[1]):
#   plt.title(list(features.columns)[j+1])
#   for i in range(0, test_size):
#     color = "r+" if y[i] == 1 else "bo"
#     plt.plot(x[i][j], i, color)
#   plt.show()

