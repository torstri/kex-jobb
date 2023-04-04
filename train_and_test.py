import cv2
import numpy as np
from skimage import measure
import get_config as conf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def print_statistics(y_test, y_pred):
  print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
  print("Balanced accurancy: ", metrics.balanced_accuracy_score(y_test, y_pred))
  print("Precision: ", metrics.precision_score(y_test, y_pred))
  print("Recall: ", metrics.recall_score(y_test, y_pred))
  print("F1: ", metrics.f1_score(y_test, y_pred))


def create_train_and_test_SVM(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=79) # 90% training and 10% test
  clf = svm.SVC(C=100 ,kernel='rbf', class_weight={1: 5}) # Linear Kernel
  clf.fit(x_train, y_train)
  curr_time = timer()
  print("Time after fitting: ", curr_time - start)
  y_pred = clf.predict(x_test)

  print(y_pred)
  print_statistics(y_test, y_pred)

def create_train_and_test_RF(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=79) # 90% training and 10% test
  clf = RandomForestClassifier(max_depth=3, random_state=0, class_weight={1: 8}, n_estimators=200)
  clf.fit(x_train, y_train)
  curr_time = timer()
  print("Time after fitting: ", curr_time - start)
  y_pred = clf.predict(x_test)

  print(y_pred)
  print_statistics(y_test, y_pred)

def create_train_and_test_KNN(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=79) # 90% training and 10% test
  clf = KNeighborsClassifier(50, weights='distance')
  clf.fit(x_train, y_train)
  curr_time = timer()
  print("Time after fitting: ", curr_time - start)
  y_pred = clf.predict(x_test)

  print(y_pred)
  print_statistics(y_test, y_pred)



test_size = 9800  # efter 34178 är det whack
start = timer()

x = np.load("./dataset/features.npy")
y = np.load("./dataset/archive-2/GroundTruth.npy")[:test_size]

curr_time = timer()

print("Loaded in after ", curr_time - start)

# create_train_and_test_SVM(x, y)
# create_train_and_test_RF(x, y)
create_train_and_test_KNN(x, y)

end = timer()
print("\nTime for execution: ", end - start)