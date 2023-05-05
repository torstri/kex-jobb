# import cv2
import numpy as np

sift_features = np.load('./SIFT/array_data/x.npy')
sift_used = np.load('./SIFT/array_data/used.npy')
abcd_features = np.load('./dataset/balanced/features.npy')
abcd_used = np.load('./dataset/balanced/used_imgs.npy')

new_features = np.zeros((3124, 169))

print(sift_used.shape)
print(abcd_used.shape)
diff = np.setdiff1d(abcd_used, sift_used)
# removed = np.delete(abcd_features, diff, 0)

result = np.concatenate((sift_features,abcd_features),axis=1)
print(result.shape)

np.save('./dataset/balanced/allFeatures.npy', result)