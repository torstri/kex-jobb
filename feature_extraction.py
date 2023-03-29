import cv2
from border_detection import border_detection
import dullrazor
# import dullrazor
from skimage import measure
from melanoma_classifier.src import _get_tabular_dataframe_utils as tab

original_path = "./orginal.png"
segemented_path =  "./segmenterad.png"

# original_img = cv2.imread(original_path)

# Dullrazor
original_img = dullrazor.dullrazor(original_path)

contours = border_detection.find_border(segemented_path, original_path)
# print(contours)

segmented_img = cv2.imread(segemented_path)
imgseg = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255.
segment_label = measure.label(imgseg)
props = measure.regionprops_table(segment_label, properties=['area', 'extent', 'perimeter', 'solidity', 'major_axis_length', 'minor_axis_length', 'centroid'])
# Retrieve assymmetry features
A1, A2 = tab.getAsymmetry(imgseg, props['centroid-0'][0], props['centroid-1'][0], props['area'][0])

# print(props)
print("A1: ", A1, ", A2: ", A2)

# Colour features extraction
colour_features = tab.getColorFeatures(original_img, imgseg)
print("Colour features: " , colour_features)

# Border feature extraction
irA = props['perimeter'][0] / props['area'][0]
irB = tab.getBorderIrregularity(props['perimeter'][0], props['minor_axis_length'][0], props['major_axis_length'][0])

print('irA: ' , irA)

print('irB: ' , irB)
