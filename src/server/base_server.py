import torch
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import numpy as np
from typing import Iterator

from src.client.base_client import BaseClient

class BaseServer:
    def __init__(
        self,
        global_epochs:int,
        local_epochs:int,
        dataset:str,
        device:torch.device,
        local_lr:float,
        global_lr:float,
        algorithm:torch.optim.Optimizer,
        model:torch.nn.Module,
        num_training_clients:int,
        training_loaders:list[DataLoader],
        testing_loader:list[DataLoader]
    ) -> None:
        # init server param
        self.global_epochs:int = global_epochs
        self.dataset:str = dataset
        self.device:torch.device = device
        self.global_lr:float = global_lr
        self.algorithm = algorithm
        self.model:torch.nn.Module = model
        self.num_activated_clients:int = num_training_clients # clients per round
        self.train_log = {'losses':{}, 'std_losses':{}, 'accs':{}, 'std_accs':{}}
        self.test_log = {'losses':{}, 'std_losses':{}, 'accs':{}, 'std_accs':{}}

        # init training and testing client
        self.training_clients:list[BaseClient] = []
        self.testing_clients:list[BaseClient] = []
        self.selected_clients:list[BaseClient] = []

        # init data for client
        for loader in training_loaders:
            tmp_client = BaseClient(local_epochs, local_lr, loader, self.model)
            self.training_clients.append(tmp_client)

        for loader in testing_loader:
            tmp_client = BaseClient(local_epochs, local_lr, loader, self.model)
            self.testing_clients.append(tmp_client)

    def _add_param(self, params:Iterator[Parameter], ratio:float):
        for global_p, local_p in zip(self.model.parameters(), params):
            global_p.data += local_p.data.clone()*ratio

    def _aggregate(self):
        total_sample = 0
        for client in self.selected_clients:
            total_sample += len(client.loader.dataset)

        for p in self.model.parameters():
            p.data = torch.zeros_like(p.data)
        for client in self.selected_clients:
            self._add_param(client.model.parameters(), len(client.loader.dataset)/total_sample)

    def compute_mean_std(self, array:list):
        return np.mean(array), np.std(array)

    def test(self):
        self._distribute_model(self.testing_clients)
        losses = []
        accs = []

        for client in self.testing_clients:
            loss, acc = client.test()
            losses.append(loss)
            accs.append(acc)

        self.test_log['losses'][round], self.test_log['std_losses'][round] = self.compute_mean_std(losses)
        self.test_log['accs'][round], self.test_log['std_accs'][round] = self.compute_mean_std(accs)

    def _distribute_model(self, list_clients:list[BaseClient]):
        for client in list_clients:
            client._set_param(self.model)

    def _train_step(self, round:int):
        losses = []
        accs = []
        for client in self.selected_clients:
            loss, acc = client.train()
            losses.append(loss)
            accs.append(acc)

        self.train_log['losses'][round] = sum(losses)/len(losses)
        self.train_log['accs'][round] = sum(accs)/len(accs)

        self._aggregate()

    def train(self):
        for r in range(self.global_epochs):
            # train global model using a batch of clients
            self.selected_clients:list[BaseClient] = np.random.choice(self.training_clients, self.num_activated_clients, replace=False)
            self._distribute_model(self.selected_clients)
            self._train_step(r)

            # evaluate global model each 20 rounds
            if (r+1)%20 == 0:
                self.test()
