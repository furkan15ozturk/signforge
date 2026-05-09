import gzip
import pickle

import torch
from torch.utils.data import Dataset


class SignTranslationDataset(Dataset):
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {**self.data[idx], 'idx': idx}



def load_dataset_file(filename):
    filename = str(filename)
    if filename.endswith(".pt"):
        return torch.load(filename, map_location="cpu")
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)