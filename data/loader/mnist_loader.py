import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

class MnistDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()

        file_ext = file_path.split('.')[-1]
        if file_ext == 'json':
            fi = open(file_path, 'r')
            data_tmp: dict = json.load(fi)
            self.X = torch.cat([torch.Tensor(data_tmp[key]) for key in data_tmp.keys()])
            self.y = torch.cat([torch.full((len(data_tmp[key]), 1), int(key)) for key in data_tmp.keys()])

        elif file_ext == 'csv':
            data_tmp = pd.read_csv(file_path).values
            self.X = torch.Tensor(data_tmp[:,1:])
            self.y = torch.Tensor(data_tmp[:,0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        ''' return torch.Size([728]), torch.Size([1])'''
        return self.X[index], self.y[index]


def get_loader(file_path, batch_size=32, shuffle=True) -> DataLoader:
    dataset = MnistDataset(file_path)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
