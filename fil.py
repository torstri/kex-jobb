import cv2
from border_detection import border_detection
# import dullrazor
from skimage import measure

original_path = "./orginal.png"
segemented_path =  "./segmenterad.png"

contours = border_detection.find_border(segemented_path, original_path)
print(contours)

original_img = cv2.imread(original_path)
segmented_img = cv2.imread(segemented_path)

imgseg = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255.

segment_label = measure.label(imgseg)

props = measure.regionprops_table(segment_label, properties=['area', 'extent', 'perimeter', 'solidity', 'major_axis_length', 'minor_axis_length', 'centroid'])

print(props)