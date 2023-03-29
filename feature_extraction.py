import os
import cv2
from border_detection import border_detection
import dullrazor
# import dullrazor
from skimage import measure
from melanoma_classifier.src import _get_tabular_dataframe_utils as tab
# import pandas
from sklearn.datasets import make_classification
import numpy as np
import csv


def get_props(segment_label):
  return measure.regionprops_table(segment_label, properties=['area', 'extent', 'perimeter', 'solidity', 'major_axis_length', 'minor_axis_length', 'centroid'])

# Retrieve assymmetry features
def get_asymmetry_features(props, imgseg):
  A1, A2 = tab.getAsymmetry(imgseg, props['centroid-0'][0], props['centroid-1'][0], props['area'][0])
  return [A1, A2]


# Colour features extraction
def get_colour_features(original_img, imgseg):
    colour_features = tab.getColorFeatures(original_img, imgseg)
    return colour_features

# Border feature extraction
def get_border_features (props):
    irA = props['perimeter'][0] / props['area'][0]
    irB = tab.getBorderIrregularity(props['perimeter'][0], props['minor_axis_length'][0], props['major_axis_length'][0])

    # print('irA: ' , irA)
    # print('irB: ' , irB)
    return [irA, irB]

def feature_extraction(original_path, segmented_path):
    original_img = dullrazor.dullrazor(original_path) # dullrazor

    contours = border_detection.find_border(segmented_path, original_path)

    segmented_img = cv2.imread(segmented_path)
    imgseg = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255.
    segment_label = measure.label(imgseg)
    props = get_props(segment_label)
    
    colour_features = get_colour_features(original_img, imgseg)
    asymmetry_features = get_asymmetry_features(props, imgseg)
    border_features = get_border_features(props)
    
    return colour_features, asymmetry_features, border_features


# original_path = "./orginal.png"
# segmented_path =  "./segmenterad.png"
# color_f, asymm_f, bord_f = feature_extraction(original_path, segmented_path)
# print("\ncolor features: ", color_f, "\n\n asymmetry features: ", asymm_f, "\n\n border features: ", bord_f)

def create_array(path):
  # x = np.array()
  y = np.zeros()
  
  imgs_path = path + "/images"
  seg_imgs_path = path + "/masks"

  for img_number in range(24306, 34320):  # 34320
    img_path = imgs_path + "/ISIC_00" + str(img_number) + ".jpg"
    seg_img_path = seg_imgs_path + "/ISIC_00" + str(img_number) + "_segmentation.jpg"
    features = feature_extraction(img_path, seg_img_path)
    
  

    
    # img = cv2.imread(img_path)
    # cv2.imshow(str(img_number), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def read_csv(path):
  x = np.zeros(10015)
  i = 0
  with open(path, 'r') as file:
    next(file)
    csvreader = csv.reader(file)
    for row in csvreader:
      if(row[1] == 1.0):
        x[i] = 1
      else:
        x[i] = 0
      i += 1
  print(x)
  
read_csv("./dataset/archive-2/GroundTruth.csv")

# create_array("./dataset/archive-2")