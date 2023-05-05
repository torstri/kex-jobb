import cv2
import numpy as np

# def construct_array(test_size):
#   data = []
#   for i in range(1, 1+test_size):
#     img = cv2.imread('./../dataset/balanced/images/' + str(i) + '.jpg')
#     data.append(img)
#   return data

# test_size = 3297

# features = []
# sift = cv2.xfeatures2d.SIFT_create()
# for i in range(1, 1+test_size):
#   img = cv2.imread('./../dataset/balanced/images/' + str(i) + '.jpg')
#   mask = cv2.imread('./../dataset/balanced/segmentations/' + str(i) + '.jpg')
#   mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#   _, descriptors = sift.detectAndCompute(img, mask)
#   features.append(descriptors)
#   print(descriptors)
#   print(len(descriptors))

# features = np.array(features)
# print(type(features))

# # data = construct_array(test_size)
# # image_descriptors = extract_sift_features(data)

def main(thresh):

  def calc_features(img, seg, th):
    sift = cv2.xfeatures2d.SIFT_create(th)
    _, des = sift.detectAndCompute(img, seg)
    return des
  
  features = []
  used_images = []

  abcd_used_images = np.load('./../dataset/balanced/used_imgs.npy')

  for i in abcd_used_images:
    img = cv2.imread('./../dataset/balanced/images/' + str(i) + '.jpg')
    mask = cv2.imread('./../dataset/balanced/segmentations/' + str(i) + '.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_des = calc_features(img, mask, thresh)
    if img_des is not None:
      features.append(img_des)
      used_images.append(i)
  features = np.vstack(features)

  k = 150
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
  flags = cv2.KMEANS_RANDOM_CENTERS
  _, labels, centres = cv2.kmeans(features, k, None, criteria, 10, flags)

  def bag_of_features(features, centres, k = 500):
    vec = np.zeros((1, k))
    for i in range(features.shape[0]):
        feat = features[i]
        diff = np.tile(feat, (k, 1)) - centres
        dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
        idx_dist = dist.argsort()
        idx = idx_dist[0]
        vec[0][idx] += 1
    return vec

  labels = []
  vec = []
  for i in abcd_used_images:
    img = cv2.imread('./../dataset/balanced/images/' + str(i) + '.jpg')
    mask = cv2.imread('./../dataset/balanced/segmentations/' + str(i) + '.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_des = calc_features(img, mask, thresh)
    if img_des is not None:
      img_vec = bag_of_features(img_des, centres, k)
      vec.append(img_vec)
      x = 1 if i > 1799 else 0
      labels.append(x)
  vec = np.vstack(vec)
  labels = np.array(labels)

  np.save('./array_data/x.npy', vec)
  np.save('./array_data/y.npy', labels)
  np.save('./array_data/used.npy', used_images)
  print(vec.shape)
  print(labels.shape)
  print(used_images)
  print(len(used_images))

main(10)