import cv2
from border_detection import border_detection
import dullrazor
# import dullrazor
from skimage import measure
from melanoma_classifier.src import _get_tabular_dataframe_utils as tab
import pandas


def get_props(segment_label):
  return measure.regionprops_table(segment_label, properties=['area', 'extent', 'perimeter', 'solidity', 'major_axis_length', 'minor_axis_length', 'centroid'])

# Retrieve assymmetry features
def get_asymmetry_features(props, imgseg):
  A1, A2 = tab.getAsymmetry(imgseg, props['centroid-0'][0], props['centroid-1'][0], props['area'][0])
  return [A1, A2]


# Colour features extraction
def get_colour_features(original_img, imgseg):
    colour_features = tab.getColorFeatures(original_img, imgseg)
    return colour_features

# Border feature extraction
def get_border_features (props):
    irA = props['perimeter'][0] / props['area'][0]
    irB = tab.getBorderIrregularity(props['perimeter'][0], props['minor_axis_length'][0], props['major_axis_length'][0])

    # print('irA: ' , irA)
    # print('irB: ' , irB)
    return [irA, irB]

def feature_extraction(original_path, segmented_path):
    original_img = dullrazor.dullrazor(original_path) # dullrazor

    contours = border_detection.find_border(segmented_path, original_path)

    segmented_img = cv2.imread(segmented_path)
    imgseg = cv2.cvtColor(segmented_img.astype('uint8'), cv2.COLOR_BGR2GRAY)/255.
    segment_label = measure.label(imgseg)
    props = get_props(segment_label)
    
    colour_features = get_colour_features(original_img, imgseg)
    asymmetry_features = get_asymmetry_features(props, imgseg)
    border_features = get_border_features(props)
    
    return colour_features, asymmetry_features, border_features


# original_path = "./orginal.png"
# segmented_path =  "./segmenterad.png"
# color_f, asymm_f, bord_f = feature_extraction(original_path, segmented_path)
# print("\ncolor features: ", color_f, "\n\n asymmetry features: ", asymm_f, "\n\n border features: ", bord_f)


def uh():
    df = pandas.DataFrame(columns=["img_id", "mask_id", "A1", "A2", "irA", "irB", "F4","F5","F6","F10","F11","F12","F13","F14","F15"])
    original_path = "./orginal.png"
    segmented_path =  "./segmenterad.png"
    features = feature_extraction(original_path,segmented_path)
    print(features)
    
uh()