import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



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

class ContrasiveLearningDataset(Dataset):
    def __init__(self, feat, feat_shadow, label, label_shadow, transform=None):
        self.feat = feat
        self.label = label

        self.feat_shadow = feat_shadow
        self.label_shadow = label_shadow

        self.transform = transform
        self.idxs = torch.arange(len(sequences))

    def __len__(self):
        return self.feat.shape[0].item()

    def __getitem__(self, idx):
        idx = self.idxs[idx]

        feat = self.feat[idx]
        label = self.label[idx]

        feat_shadow = self.feat_shadow[idx]
        label_shadow = self.label_shadow[idx]

        return (feat, label), (feat_shadow, label_shadow)
    
    def randomize(self):
        np.random.shuffle(self.idxs)
    
    def reset(self):
        self.idxs = torch.arange(len(self.feat))

class TemporalDataset(Dataset):
    def __init__(self, feat, label, transform=None):
        self.feat = feat
        self.label = label

        self.transform = transform
        self.idxs = torch.arange(len(sequences))

    def __len__(self):
        return self.feat.shape[0].item()

    def __getitem__(self, idx):
        idx = self.idxs[idx]

        feat = self.feat[idx]
        label = self.label[idx]

        return (feat, label)

    def randomize(self):
        np.random.shuffle(self.idxs)
    
    def reset(self):
        self.idxs = torch.arange(len(self.feat))


if __name__ == "__main__":
    pose_dict_path = "../data/train.npy"
    sequences = DataUtils.read_npy(pose_dict_path)
    DataUtils.build_temporal_sequence(sequences, 10, 10, 1)
