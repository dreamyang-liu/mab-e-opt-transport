from tensorflow import keras
import tensorflow as tf
import keras
from keras.models import Sequential
import numpy as np



class MABe_Data_Generator(keras.utils.Sequence):
    def __init__(self,  pose_dict, 
                        batch_size = 2,
                        feature_dimensions=(2,2,7),
                        classes=['other', 'investigation', 'attack', 'mount'],
                        past_frames=100,
                        future_frames=100,
                        frame_skip=1,
                        shuffle=True):

        self.batch_size = batch_size
        self.feature_dimensions = feature_dimensions

        self.classes=classes
        self.n_classes=len(classes)

        self.past_frames = past_frames
        self.future_frames = future_frames
        self.frame_skip = frame_skip

        # Data is arranged as [t, flattened_feature_dimensions]
        #        where t => [past_frames + 1 + future_frames]
        self.dim = (
                (past_frames + 1 + future_frames),
                np.prod(self.feature_dimensions)
            )

        self.shuffle = shuffle
        
        # Raw Data Containers
        self.X = {}
        self.y = []

        # Load raw pose dictionary
        self.load_pose_dictionary(pose_dict)
        
        # Setup Utilities
        self.setup_utils()

        # Preprocess annotations
        self.preprocess_annotations()

        # Epoch End preparations
        self.on_epoch_end()


    def load_pose_dictionary(self, pose_dict):
        # Load raw pose dictionary
        self.pose_dict = pose_dict
        self.video_keys = list(pose_dict.keys())

    def setup_utils(self):
        # Set up padding utilities
        self.setup_padding_utils()

        # Setup class conversion utils
        self.setup_class_conversion_utils()

    def setup_padding_utils(self):
        # Prepare to pad frames
        self.left_pad = self.past_frames * self.frame_skip
        self.right_pad = self.future_frames * self.frame_skip
        self.pad_width = (self.left_pad, self.right_pad), (0, 0), (0, 0), (0, 0)

    def setup_class_conversion_utils(self):
        # Prepare a classname to idx map
        self.classname_to_index_map = {}
        for _idx, class_name in enumerate(self.classes):
            self.classname_to_index_map[class_name] = _idx
    
    def classname_to_index(self, annotations_list):
        """
        Converts a list of string classnames into numeric indices
        """
        return np.vectorize(self.classname_to_index_map.get)(annotations_list)

    def preprocess_annotations(self):
        # Define arrays to map video keys to frames 
        self.video_indices = []
        self.frame_indices = []

        self.action_annotations = []

        for video_index, video_key in enumerate(self.video_keys):
            annotations = self.pose_dict[video_key]['annotations']
            self.action_annotations.extend(annotations) # add annotations to action_annotations        

            number_of_frames = len(annotations)

            self.video_indices.extend([video_index] * number_of_frames) # Keep a record of video_indices
            self.frame_indices.extend(range(number_of_frames)) # Keep a record of frame indices
            self.X[video_key] = np.pad(self.pose_dict[video_key]['keypoints'], self.pad_width) # Add padded keypoints for each video key

        self.y = self.classname_to_index(self.action_annotations) # convert text labels to indices
        self.X_dtype = self.X[video_key].dtype # Store D_types of X 
        
        #generate a global index list for all data points
        self.indices = np.arange(len(self.frame_indices)) 

    def __len__(self):
        return len(self.indices) // self.batch_size

    def get_X(self, data_index):
        """
        Obtains the X value from a particular 
        global index
        """
        # Obtain video key for this datapoint
        video_key = self.video_keys[
            self.video_indices[data_index]
        ]
        # Identify the (local) frame_index
        frame_index = self.frame_indices[data_index]
        # Slice from beginning of past frames to end of future frames 
        slice_start_index = frame_index - self.left_pad
        slice_end_index = frame_index + 1 + self.right_pad 
        assert slice_start_index >= 0
        _X = self.X[video_key][
            slice_start_index:slice_end_index:self.frame_skip
        ]
        return _X

        
    def __getitem__(self, index):
        batch_size = self.batch_size
        batch_indices = self.indices[
                        index*batch_size:(index+1)*batch_size]
        
        X = np.empty((batch_size, *self.dim), self.X_dtype)
        
        for batch_index, data_index in enumerate(batch_indices):
            # Obtain the post-processed X value at the said data index
            _X = self.get_X(data_index)
            # Reshape the _X to the expected dimensions
            X[batch_index] = np.reshape(_X, self.dim)

        y_vals = self.y[batch_indices]
        # Converting to one hot because F1 callback needs one hot
        y = keras.utils.to_categorical(y_vals, num_classes=self.n_classes)

        return X, y


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)


if __name__ == "__main__":

    pose_dict_path = "../datasets/mabe_task1_data.npy"
    pose_dict = np.load(pose_dict_path, allow_pickle=True).item()

    generator = MABe_Data_Generator(
                    pose_dict,
                    batch_size=2,
                    feature_dimensions=(2,2,7),
                    classes=['other', 'investigation', 'attack', 'mount'],
                    past_frames=100,
                    future_frames=100,
                    frame_skip=1,
                    shuffle=True)
    
    print(generator[0])
    print("Length : ", len(generator))