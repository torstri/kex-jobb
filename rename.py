import os
import cv2
import numpy as np

# for i in range(24306, 34320):
#   old_path = "./dataset/archive-2/masks/ISIC_00" + str(i) + "_segmentation.png"
#   new_path = "./dataset/archive-2/masks/ISIC_00" + str(i) + ".jpg"
#   os.rename(old_path, new_path)

# for i in range(24306, 34321):
#     path = "./dataset/archive-2/masks/ISIC_00" + str(i) + ".jpg"
#     im = cv2.imread(path)
#     im = cv2.resize(im, (224, 224))
#     cv2.imwrite(path, im)
#     if i % 1000 == 0:
#       print(i)

# Snygg funktion
def switch_files(number1, number2):
  # old1_path = "./dataset/archive-2/masks/ISIC_00" + str(number1) + ".jpg"
  # old2_path = "./dataset/archive-2/masks/ISIC_00" + str(number2) + ".jpg"
  # temp ="./dataset/archive-2/masks/ISIC_0039000.jpg"
  old1im_path = "./dataset/archive-2/images/ISIC_00" + str(number1) + ".jpg"
  old2im_path = "./dataset/archive-2/images/ISIC_00" + str(number2) + ".jpg"
  tempim ="./dataset/archive-2/images/ISIC_0039000.jpg"
  old1s_path = "./dataset/archive-2/segmentations/ISIC_00" + str(number1) + ".jpg"
  old2s_path = "./dataset/archive-2/segmentations/ISIC_00" + str(number2) + ".jpg"
  temps ="./dataset/archive-2/segmentations/ISIC_0039000.jpg"

  # os.rename(old1_path, temp)
  # os.rename(old2_path, old1_path)
  # os.rename(temp, old2_path)

  os.rename(old1im_path, tempim)
  os.rename(old2im_path, old1im_path)
  os.rename(tempim, old2im_path)

  os.rename(old1s_path, temps)
  os.rename(old2s_path, old1s_path)
  os.rename(temps, old2s_path)

  tempy = y[number1-24306]
  y[number1-24306] = y[number2-24306]
  y[number2-24306] = tempy

y = np.load("./dataset/archive-2/GroundTruth.npy")

# k = 34320
# i = 24306
# while i < k:
#   img = cv2.imread("./dataset/archive-2/segmentations/ISIC_00" + str(i) + ".jpg")
#   if img.sum() == 0:
#     print("found! Files switching: ", str(i), " and ", str(k))
#     switch_files(i, k)
#     k -= 1
#   else:
#     i += 1

# switch_files(32174, 34179)

np.save("./dataset/archive-2/GroundTruth.npy", y)