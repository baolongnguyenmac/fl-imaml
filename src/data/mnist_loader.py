import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

class MnistDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()

        fi = open(file_path, 'r')
        data_tmp: dict = json.load(fi)
        self.X:torch.Tensor = torch.cat([torch.Tensor(data_tmp[key]) for key in data_tmp.keys()])
        self.y:torch.Tensor = torch.cat([torch.full((len(data_tmp[key]),), int(key)) for key in data_tmp.keys()])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        # return (torch.Size([728]), torch.Size([1]))
        return self.X[index], self.y[index].long()


def get_loader(file_path: str, batch_size: int=32, shuffle: bool=True) -> DataLoader:
    dataset = MnistDataset(file_path)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
