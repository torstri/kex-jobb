import cv2
import numpy as np
import csv

def read_csv(path):
  x = np.zeros(10016)
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

y = read_csv("./dataset/archive-2/GroundTruth.csv")

print(y.shape)

np.save("./dataset/archive-2/GroundTruth.npy", y)