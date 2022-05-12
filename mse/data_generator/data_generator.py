from builtins import breakpoint
import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABCMeta, abstractmethod




class DataUtils:

    @staticmethod
    def read_npy(file_path, flatten=False):
        data_dict = np.load(file_path, allow_pickle=True).item()
        sequences_keys = data_dict['sequences'].keys()
        sequences_values = data_dict['sequences']
        sequences = {}
        for key in sequences_keys:
            feat = torch.from_numpy(sequences_values[key]['keypoints']).float()
            label = torch.from_numpy(sequences_values[key]['annotations']).long()
            if flatten:
                feat = feat.view(feat.shape[0], -1)
            sequences[key] = {
                'feat': feat,
                'label': label,
            }
        return sequences
    
    @staticmethod
    def build_single_contrasive_learning_sequence(sequence):
        sequence_feature = sequence['feat'][:-1, ...]
        sequence_feature_shadow = sequence['feat'][1:, ...]

        sequence_label = sequence['label'][:-1]
        sequence_label_shadow = sequence['label'][1:]

        return {
            'feat': sequence_feature,
            'label': sequence_label,
            'feat_shadow': sequence_feature_shadow,
            'label_shadow': sequence_label_shadow,
        }
    
    @staticmethod
    def build_contrasive_learning(sequences):
        feats = []
        labels = []

        feats_shadow = []
        labels_shadow = []

        for _, sequence in sequences.items():
            sequence_contrasive = DataUtils.build_single_contrasive_learning_sequence(sequence)
            feats.append(sequence_contrasive['feat'])
            labels.append(sequence_contrasive['label'])

            feats_shadow.append(sequence_contrasive['feat_shadow'])
            labels_shadow.append(sequence_contrasive['label_shadow'])

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        
        feats_shadow = torch.cat(feats_shadow, dim=0)
        labels_shadow = torch.cat(labels_shadow, dim=0)
        return feats, labels, feats_shadow, labels_shadow

    @staticmethod
    def build_single_temporal_sequence(sequence, past_frames=0, future_frames=0, frame_skip=1):
        sequence_feature = sequence['feat']
        sequence_label = sequence['label']

        sample_number = sequence_feature.shape[0] - past_frames - future_frames
        temporal_feature = torch.tensor(torch.ones(sample_number, 1 + past_frames + future_frames, *sequence_feature.shape[1:])).float()
        temporal_label = torch.tensor(torch.ones(sample_number)).long()

        for i in range(past_frames, sample_number + past_frames):
            temporal_feature[i - past_frames, ...] = sequence_feature[i - past_frames:i + future_frames+1, ...].clone()
            temporal_label[i - past_frames] = sequence_label[i]

        return {
            'feat': temporal_feature,
            'label': temporal_label,
        }
    
    @staticmethod
    def build_temporal_sequence(sequences, past_frames=0, future_frames=0, frame_skip=1):
        feats = []
        labels = []
        for _, sequence in sequences.items():
            sequence_temporal = DataUtils.build_single_temporal_sequence(sequence, past_frames, future_frames, frame_skip)
            feats.append(sequence_temporal['feat'])
            labels.append(sequence_temporal['label'])
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels
    
    @staticmethod
    def build_nontemporal_single_sequence(sequence):
        sequence_feature = sequence['feat']
        sequence_label = sequence['label']

        return type('data', (object, ), {
            'feat': sequence_feature,
            'label': sequence_label,
        })
    
    @staticmethod
    def build_nontemporal_sequence(sequences):
        feats = []
        labels = []
        for _, sequence in sequences.items():
            sequence_nontemporal = DataUtils.build_nontemporal_single_sequence(sequence)
            feats.append(sequence_nontemporal.feat)
            labels.append(sequence_nontemporal.label)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels
    
    @staticmethod
    def augment_fn(to_augment):
        """ 
        Augment sequences
            * Rotation - All frames in the sequence are rotated by the same angle
                using the euler rotation matrix
            * Shift - All frames in the sequence are shifted randomly
                but by the same amount
        """
        if len(to_augment.shape) != 4:
            x = to_augment[:, :28].reshape(-1, 2, 7, 2)
        else:
            x = to_augment

        # Rotate
        angle = (np.random.rand()-0.5) * (np.pi * 2)
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        x = np.dot(x, rot)

        # Shift - All get shifted together
        shift = (np.random.rand(2)-0.5) * 2 * 0.25
        x = x + shift

        if len(to_augment.shape) != 4:

            x = np.concatenate([x.reshape(-1, 28), to_augment[:, 28:]], axis = -1)

        return x

class MyDataset(Dataset):
    
    @abstractmethod
    def optimize(self, optimizer):
        pass
class ContrasiveLearningDataset(MyDataset):
    def __init__(self, path, args, transform=None):
        self.args = args
        data = DataUtils.read_npy(path, flatten=self.args.flatten)
        data = DataUtils.build_contrasive_learning(data)
        self.feat, self.label, self.feat_shadow, self.label_shadow = data


        self.transform = transform
        self.idxs = torch.arange(self.feat.shape[0])
        
        self.prepare_batch_idxs()
    
    def prepare_batch_idxs(self):
        self.batch_idxs = []
        for i in range(0, self.idxs.shape[0], self.args.batch_size):
            self.batch_idxs.append([i, min(i + self.args.batch_size, self.idxs.shape[0])])
        self.batch_idxs = torch.tensor(self.batch_idxs)

    def __len__(self):
        return self.batch_idxs.shape[0]

    def __getitem__(self, idx):
        idx = self.idxs[self.batch_idxs[idx][0]:self.batch_idxs[idx][1]]

        feat = self.feat[idx]
        label = self.label[idx]

        feat_shadow = self.feat_shadow[idx]
        label_shadow = self.label_shadow[idx]

        return (feat, label), (feat_shadow, label_shadow)
    
    def get_input_dim(self):
        return self.feat.shape[1:]

    def get_class_dim(self):
        return self.label.unique().shape[0]
    
    def optimize(self, optimizer=None):
        raise NotImplementedError("optimize method is not implemented")
    
    def randomize(self):
        np.random.shuffle(self.idxs)
    
    def reset(self):
        self.idxs = torch.arange(len(self.feat))

class NonTemporalDataset(MyDataset):

    def __init__(self, path, args, transform=None):
        self.path = path
        self.args = args
        self.transform = transform
        data = DataUtils.read_npy(path, flatten=self.args.flatten)
        self.feat, self.label = DataUtils.build_nontemporal_sequence(data)
        self.raw_label = copy.deepcopy(self.label)
        self.idxs = torch.arange(self.feat.shape[0])
        self.prepare_batch_idxs()
    
    def prepare_batch_idxs(self):
        self.batch_idxs = []
        for i in range(0, self.idxs.shape[0], self.args.batch_size):
            self.batch_idxs.append([i, min(i + self.args.batch_size, self.idxs.shape[0])])
        self.batch_idxs = torch.tensor(self.batch_idxs)

    def __len__(self):
        return self.batch_idxs.shape[0]
    
    def __getitem__(self, idx):
        idx = self.idxs[self.batch_idxs[idx][0]:self.batch_idxs[idx][1]]

        feat = self.feat[idx]
        label = self.label[idx]

        return feat, label
    
    def get_input_dim(self):
        return self.feat.shape[1:]

    def get_class_dim(self):
        return self.label.unique().shape[0]
    
    def optimize(self, optimizer, payload=None):
        if optimizer is None:
            print("Optimizer is not set, use default label")
        else:
            new_label = optimizer.optimize(payload)
            self.label = new_label
            
    def randomize(self):
        np.random.shuffle(self.idxs)
    
    def reset(self):
        self.idxs = torch.arange(len(self.feat))

class TemporalDataset(MyDataset):
    def __init__(self, feat, label, transform=None):
        self.feat = feat
        self.label = label

        self.transform = transform
        self.idxs = torch.arange(feat.shape[0])

    def __len__(self):
        return self.feat.shape[0].item()

    def __getitem__(self, idx):
        idx = self.idxs[idx]

        feat = self.feat[idx]
        label = self.label[idx]

        return (feat, label)
    
    def optimize(self, optimizer, payload=None):
        if optimizer is None:
            print("Optimizer is not set, use default label")
        else:
            new_label = optimizer.optimize(self.label)
            self.label = new_label

    def randomize(self):
        np.random.shuffle(self.idxs)
    
    def reset(self):
        self.idxs = torch.arange(len(self.feat))


if __name__ == "__main__":
    pass
