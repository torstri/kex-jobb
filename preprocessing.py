import numpy as np
import cv2 as cv
img = cv.imread('ISIC_0024308_segmentation.png')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)
cv.imshow('sift_keypoints.jpg',img)


cv.waitKey()
cv.destroyAllWindows()