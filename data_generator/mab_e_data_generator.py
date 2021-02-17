from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
import numpy as np


class_to_number = {'other': 0,'investigation': 1,
                   'attack' : 2, 'mount' : 3}

def text_to_num(anno_list):
  return np.vectorize(class_to_number.get)(anno_list)

class MABe_Data_Generator(keras.utils.Sequence):
    def __init__(self,  pose_dict_path, 
                        batch_size = 2,
                        config = {
                            "dimensions" : None,
                            "max_classes" : 4,
                            "past_frames" : 100,
                            "future_frames" : 100, 
                            "frame_skip" : 1
                        },
                        shuffle=True):
        
        self.pose_dict = np.load(pose_dict_path, allow_pickle=True).item()
        self.video_keys = list(self.pose_dict.keys())

        self.batch_size = batch_size
        self.dimensions = config["dimensions"] 
        
        self.max_classes=config["max_classes"]
        self.past_frames = config["past_frames"] 
        self.future_frames = config["future_frames"] 
        self.frame_skip = config["frame_skip"] 

        self.shuffle = shuffle
        
        self.video_indexes = []
        self.frame_indexes = []
        self.X = {}
        self.y = []
        self.pad = self.past_frames * self.frame_skip
        future_pad = self.future_frames * self.frame_skip
        pad_width = (self.pad, future_pad), (0, 0), (0, 0), (0, 0)
        for vc, key in enumerate(self.video_keys):
          anno = self.pose_dict[key]['annotations']
          self.y.extend(anno)
          nframes = len(anno)
          self.video_indexes.extend([vc for _ in range(nframes)])
          self.frame_indexes.extend(range(nframes))
          self.X[key] = np.pad(self.pose_dict[key]['keypoints'], pad_width)

        self.y = text_to_num(self.y)
        self.X_dtype = self.X[key].dtype
        self.indexes = np.arange(len(self.frame_indexes))
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        bs = self.batch_size
        indexes = self.indexes[index*bs:(index+1)*bs]
        X = np.empty((bs, *self.dimensions), self.X_dtype)
        y_vals = self.y[indexes]
        # Converting to one hot because F1 callback needs one hot
        y = np.zeros( (bs,self.max_classes), np.float32)
        y[np.arange(bs), y_vals] = 1
        for bi, idx in enumerate(indexes):
          vkey = self.video_keys[self.video_indexes[idx]]
          fi = self.frame_indexes[idx] + self.pad
          start = fi - self.past_frames*self.frame_skip
          stop = fi + (self.future_frames + 1)*self.frame_skip
          assert start >= 0
          Xi = self.X[vkey][start:stop:self.frame_skip]
          X[bi] = np.reshape(Xi, self.dimensions)
        
        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":

    generator = MABe_Data_Generator(
                    pose_dict_path = "../datasets/mabe_task1_data.npy",
                    batch_size=2,
                    dimensions=10,
                    max_classes = 4,
                    past_frames=0, future_frames=0, frame_skip=1, shuffle=True)
    
    for x, y  in generator:
        print(x, y)
        break
    
    print("Length : ", len(generator))