# Kanske behöver avinstallera opencv-python och köra:
# pip3 install opencv-contrib-python

# https://www.thepythoncode.com/article/sift-feature-extraction-using-opencv-in-python?utm_content=cmp-true

import cv2 

# reading the image
img = cv2.imread('./ISIC_0024306_segmentation.png')
# convert to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

nfeatures = 0
nOctaveLayers = 3
contrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6

# create SIFT feature extractor
sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

# detect features from the image
keypoints, descriptors = sift.detectAndCompute(img, None)

# draw the detected key points
sift_image = cv2.drawKeypoints(gray, keypoints, img)
# show the image
cv2.imshow('image', sift_image)
# save the image
# cv2.imwrite("newImg.jpg", sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()