import numpy as np

def load_mabe_data(data_path):
    data_dict = np.load(data_path, allow_pickle=True).item()
    dataset = data_dict['sequences']
    vocabulary = data_dict['vocabulary']
    return  dataset, vocabulary