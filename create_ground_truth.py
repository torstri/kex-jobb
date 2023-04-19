import os
import cv2
import numpy as np

# y = np.zeros(1800+1500)

# for i in range(1, 1801):
#   old_path = "./dataset/balanced/train/benign/" + str(i) + ".jpg"
#   new_path = "./dataset/balanced/" + str(i) + ".jpg"
#   os.rename(old_path, new_path)

# for i in range(2588, 3300):
#   old_path = "./dataset/balanced/" + str(i) + ".jpg"
#   new_path = "./dataset/balanced/" + str(i - 2) + ".jpg"
#   os.rename(old_path, new_path)
  # y[1799+i] = 1

y = np.load("./dataset/balanced/GroundTruth.npy")

sub = y[:-3]
print(len(sub))
np.save("./dataset/balanced/GroundTruth.npy", sub)

# np.save("./dataset/balanced/GroundTruth.npy", y)