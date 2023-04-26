import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# có thể sử dụng cho cả MNIST lẫn EMNIST (chúng nó same)
class MnistDataset(Dataset):
    def __init__(self, X:torch.Tensor, y:torch.Tensor, cid:int=None, noise:bool=False) -> None:
        super().__init__()
        self.X:torch.Tensor = X/255.
        self.y:torch.Tensor = y

        if noise:
            noise_mask = np.random.normal(0, 0.1*cid/100, self.X.shape) # num clients = 100
            self.X  += noise_mask
            self.X /= torch.max(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        # return (torch.Size([728]), torch.Size([1]))
        return self.X[index], self.y[index].long()

# có thể sử dụng cho cả CIFAR-10 lẫn CIFAR-100 (chúng nó same)
class CifarDataset(Dataset):
    def __init__(self, X:torch.Tensor, y:torch.Tensor, cid:int=None, noise:bool=False) -> None:
        super().__init__()
        self.X:torch.Tensor = X/255.
        self.y:torch.Tensor = y

        if noise:
            noise_mask = np.random.normal(0, 0.1*cid/100, self.X.shape) # num clients = 100
            self.X += noise_mask
            self.X /= torch.max(self.X)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index].reshape(3,32,32), self.y[index].long()

def get_loader(file_path: str, batch_size: int=32, shuffle: bool=True, split_supp_query:bool = True, noise:bool = False) -> DataLoader:
    dataset = MnistDataset if 'mnist' in file_path else CifarDataset if 'cifar' in file_path else None
    file_name = file_path.split('/')[-1]
    cid = int(file_name[:-5]) if 'train' in file_path else int(file_name[:-7])

    # load data from file into torch.Tensor
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
        support_set, query_set = dataset(X_support, y_support, cid, noise), dataset(X_query, y_query, cid, noise)
        return DataLoader(support_set, batch_size=batch_size, shuffle=shuffle), DataLoader(query_set, batch_size=batch_size, shuffle=shuffle)
    else:
        return DataLoader(dataset=dataset(X, y, cid, noise), batch_size=batch_size, shuffle=shuffle)
