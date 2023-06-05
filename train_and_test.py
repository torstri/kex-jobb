import cv2
import numpy as np
from skimage import measure
# import get_config as conf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from timeit import default_timer as timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def print_statistics(y_test, y_pred):
  print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
  print("Balanced accurancy: ", metrics.balanced_accuracy_score(y_test, y_pred))
  print("Precision: ", metrics.precision_score(y_test, y_pred))
  print("Recall: ", metrics.recall_score(y_test, y_pred))
  print("F1: ", metrics.f1_score(y_test, y_pred))

def create_train_and_test(x, y, clf):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) # 90% training and 10% test
  clf.fit(x_train, y_train) 
  y_pred = clf.predict(x_test)
  return y_pred, metrics.balanced_accuracy_score(y_test, y_pred)

  # print(y_pred)
  # print_statistics(y_test, y_pred)


def train_and_test(x, y, number_of_loops):
  # test_size = 3000  # efter 34178 är det whack

  clfKNN = KNeighborsClassifier(n_neighbors=8)
  clfSVM = svm.SVC(C=100 ,kernel='rbf')
  clfRF = RandomForestClassifier(n_estimators=5)
  clfNN = MLPClassifier(hidden_layer_sizes=[8], activation='relu', max_iter=100)

  total_acc = 0
  for i in range(0, number_of_loops):
    res, acc = create_train_and_test(x, y, clfNN)
    total_acc += acc
    if (i % 10 == 0):
      print(i)
  print("Balanced mean accurancy: ", total_acc/number_of_loops)
  return total_acc/number_of_loops
  
print("Loading data...")

used = np.load('./SIFT/array_data/used.npy')

x = np.load('./SIFT/array_data/x.npy')
y = np.load('./SIFT/array_data/y.npy')

# Only use the images in used images in both x and y. x is 2-dimensional and y is 1-dimensional, use the second dimension of x to get the image number
print("x shape: ", x.shape)
print("y shape: ", y.shape)
print("used:" , used)



print("Loaded in ", len(x), " images")

train_and_test(x, y, 50)

print("Loading data...")

start = timer()
curr_time = timer()
print("Loaded in after ", curr_time - start)

used = np.load('./dataset/balanced/used_imgs.npy')
print("used: ", used)
for i in range(0, len(used)):
  used[i] = used[i] - 1
x = np.load("./dataset/balanced/features.npy")
y = np.load("./dataset/balanced/GroundTruth.npy")
y = y[used]
print("y shape: ", y.shape)
print("x shape: ", x.shape)
train_and_test(x, y, 50)

end = timer()
print("\nTime for execution: ", end - start)