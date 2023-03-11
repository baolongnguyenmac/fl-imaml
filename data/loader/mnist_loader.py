import torch
from torch.utils.data import Dataset, DataLoader
import json


class MnistDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        with open(file_path, 'r') as fi:
            data_tmp: dict = json.load(fi)

        self.X = torch.cat([torch.Tensor(data_tmp[key]) for key in data_tmp.keys()])
        self.y = torch.cat(
            [torch.full((len(data_tmp[key]), 1), int(key)) for key in data_tmp.keys()])

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
