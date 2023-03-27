import numpy as np
import cv2

# s_path = "./../segmenterad.png"
# o_path = "./../orginal.png"

def find_border (segmented_path, original_path):

    img = cv2.imread(segmented_path)
    orImg = cv2.imread(original_path)

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

    # cv2.drawContours(orImg, contours, -1, (0,255,0), 3)
    # cv2.imshow("Konturer", orImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
     
# find_border(s_path, o_path)