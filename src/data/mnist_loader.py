import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

class MnistDataset(Dataset):
    def __init__(self, X:torch.Tensor, y:torch.Tensor) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        # return (torch.Size([728]), torch.Size([1]))
        return self.X[index], self.y[index].long()


def get_loader(file_path: str, batch_size: int=32, shuffle: bool=True, split_supp_query:bool=True) -> DataLoader:
    # loader data from file into torch.Tensor
    fi = open(file_path, 'r')
    data_tmp: dict = json.load(fi)
    X:torch.Tensor = torch.cat([torch.Tensor(data_tmp[key]) for key in data_tmp.keys()])
    y:torch.Tensor = torch.cat([torch.full((len(data_tmp[key]),), int(key)) for key in data_tmp.keys()])

    # create loader
    if split_supp_query:
        # shuffle X, y
        idx = list(range(len(y)))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # cut them into 2parts (support, query)
        tmp = int(len(y)*0.2)
        X_support, y_support = X[:tmp], y[:tmp]
        X_query, y_query = X[tmp:], y[tmp:]

        # return 2loaders
        support_set, query_set = MnistDataset(X_support, y_support), MnistDataset(X_query, y_query)
        return DataLoader(support_set, batch_size=batch_size, shuffle=shuffle), DataLoader(query_set, batch_size=batch_size, shuffle=shuffle)
    else:
        return DataLoader(MnistDataset(X, y), batch_size=batch_size, shuffle=shuffle)
