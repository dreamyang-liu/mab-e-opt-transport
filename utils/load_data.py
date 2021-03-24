import numpy as np


def load_mabe_data_task1(data_path):
    data_dict = np.load(data_path, allow_pickle=True).item()
    dataset = data_dict['sequences']
    vocabulary = data_dict['vocabulary']
    return dataset, vocabulary


def load_mabe_data_task2(data_path):
    data_dict = np.load(data_path, allow_pickle=True).item()
    dataset = data_dict['sequences']
    vocabulary = data_dict['vocabulary']
    anno_ids = np.unique([s['annotator_id'] for _, s in dataset.items()])
    dataset_annotators = {}
    for annotator_id in anno_ids:
        annotator_data = {skey: seq for skey, seq in dataset.items()
                          if seq['annotator_id'] == annotator_id}
        dataset_annotators[annotator_id] = annotator_data
    return dataset_annotators, vocabulary


def load_mabe_data_task3(data_path):
    dataset = np.load(data_path, allow_pickle=True).item()
    return dataset
