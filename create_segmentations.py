import get_config as conf
import tensorflow as tf
from melanoma_classifier.src._get_segmentations_utils import *

def construct_df(test_size):
  data = []
  for i in range(1, 1+test_size):
    # result = "malignant" if y[i-24306] == 1 else "benign"
    data.append([str(i) + ".jpg"]) # , result
  df = pd.DataFrame(data, columns=["filename"]) # , "target"
  return df

df = construct_df(3297)
cfg = conf.get_config()
images_path = "./dataset/balanced/images/"
segmentations_path = "./dataset/balanced/segmentations/"

compute_segmentations(cfg, df, images_path, segmentations_path)