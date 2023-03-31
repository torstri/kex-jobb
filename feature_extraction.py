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
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from timeit import default_timer as timer



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
    ir_a = props['perimeter'][0] / props['area'][0]
    ir_b = tab.getBorderIrregularity(props['perimeter'][0], props['minor_axis_length'][0], props['major_axis_length'][0])
    return [ir_a, ir_b]

def feature_extraction(original_path, segmented_path):
    # original_img = dullrazor.dullrazor(original_path) # Extremt långsam. Tar 2/3 av totala tiden typ.
    original_img = cv2.imread(original_path)

    # Find contours, not used atm
    # contours = border_detection.find_border(segmented_path, original_path)

    segmented_img = cv2.imread(segmented_path)
    imgseg = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255.
    segment_label = measure.label(imgseg)
    props = get_props(segment_label)
    
    colour_features = get_colour_features(original_img, imgseg)
    asymmetry_features = get_asymmetry_features(props, imgseg)
    border_features = get_border_features(props)  
    
    return [colour_features, asymmetry_features, border_features]

def create_array(path):
  test_size = 3000
  x = np.zeros((test_size, 13)) # number of images and number of features
  y = read_csv(path + "/GroundTruth.csv")
  y = y[:test_size]
  
  imgs_path = path + "/images"
  seg_imgs_path = path + "/masks"

  for img_number in range(24306, 24306+test_size):  # 34320 is the maximum
    img_path = imgs_path + "/ISIC_00" + str(img_number) + ".jpg"
    seg_img_path = seg_imgs_path + "/ISIC_00" + str(img_number) + "_segmentation.png"
    features = feature_extraction(img_path, seg_img_path) # Bild 26042 är whack eftersom hela bilden är utslaget
    i = 0
    for feature_list in features:
      for feature in feature_list:
        x[img_number-24306][i] = feature
        i += 1

  curr_time = timer()
  print("Time for feature extraction: ", curr_time - start)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109) # 70% training and 30% test

  clf = svm.SVC(kernel='linear') # Linear Kernel

  clf.fit(x_train, y_train)

  curr_time = timer()
  print("Time after fitting: ", curr_time - start)

  y_pred = clf.predict(x_test)
  print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



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

start = timer()
create_array("./dataset/archive-2")
end = timer()
print("\nTime for execution: ", end - start)