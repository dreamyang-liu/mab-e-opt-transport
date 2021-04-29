import numpy as np

def transpose_last_axis(orig_pose_dictionary):
  for key in orig_pose_dictionary:
    X = orig_pose_dictionary[key]['keypoints']
    X = X.transpose((0,1,3,2)) #last axis is x, y coordinates
    orig_pose_dictionary[key]['keypoints'] = X
  return orig_pose_dictionary

def normalize_data(orig_pose_dictionary):
  """ 
  All sequences have 
  * Channel 0 with scale of 1024 
  * Channel 1 with scale of 570
  """
  for key in orig_pose_dictionary:
    X = orig_pose_dictionary[key]['keypoints']
    X[..., 0] = X[..., 0]/1024
    X[..., 1] = X[..., 1]/570
    orig_pose_dictionary[key]['keypoints'] = X
  return orig_pose_dictionary
