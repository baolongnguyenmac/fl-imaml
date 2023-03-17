import torch
from torch.nn.parameter import Parameter
import numpy as np
from typing import Iterator
from datetime import datetime
from os.path import isdir, join
from os import mkdir
import json

from client.base_client import BaseClient

class BaseServer:
    def __init__(
            self,
            global_epochs:int,
            # local_epochs:int,
            device:torch.device,
            # local_lr:float,
            global_lr:float,
            # algorithm:torch.optim.Optimizer,
            model:torch.nn.Module,
            num_activated_clients:int,
            # training_loaders:list[DataLoader],
            # testing_loader:list[DataLoader],
            command:dict
        ) -> None:

        # init server param
        self.global_epochs:int = global_epochs
        self.device:torch.device = device
        self.global_lr:float = global_lr
        # self.algorithm = algorithm
        self.model:torch.nn.Module = model
        self.num_activated_clients:int = num_activated_clients # clients per round
        self.command:dict = command
        self.train_log = {'losses':{}, 'std_losses':{}, 'accs':{}, 'std_accs':{}}
        self.test_log = {'losses':{}, 'std_losses':{}, 'accs':{}, 'std_accs':{}}

        # init training and testing client
        self.training_clients:list[BaseClient] = []
        self.testing_clients:list[BaseClient] = []
        self.selected_clients:list[BaseClient] = []

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

    def test(self, round:int):
        print(f'\nRun test on {len(self.testing_clients)} clients')
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
        print('Distribute global model')
        for client in list_clients:
            client._set_param(self.model)

    def _train_step(self, round:int):
        losses = []
        accs = []
        for client in self.selected_clients:
            loss, acc = client.train()
            losses.append(loss)
            accs.append(acc)

        self.train_log['losses'][round], self.train_log['std_losses'][round] = self.compute_mean_std(losses)
        self.train_log['accs'][round], self.train_log['std_accs'][round] = self.compute_mean_std(accs)

        self._aggregate()

    def train(self):
        for r in range(self.global_epochs):
            print(f'\n============= Round {r} =============\n')
            # train global model using a batch of clients
            self.selected_clients:list[BaseClient] = np.random.choice(self.training_clients, self.num_activated_clients, replace=False)
            self._distribute_model(self.selected_clients)
            self._train_step(r)
            print(f"[Train]Loss: {self.train_log['losses'][r]:>7f}, Acc: {self.train_log['accs'][r]:>7f}")

            # evaluate global model each 20 rounds
            if (r+1)%5 == 0 or r == 0:
                self.test(r)
                print(f"[Test] Loss: {self.test_log['losses'][r]:>7f}, Acc: {self.test_log['accs'][r]:>7f}")

        # save log to ./experiment
        self.save_log()

    def save_log(self):
        dir_ = join('../experiment', datetime.today().strftime('%Y-%m-%d'))
        print(f'\nWrite log to {dir_}')

        if not isdir(dir_):
            mkdir(dir_)

        dir__ = join(dir_, datetime.today().strftime('%H:%M'))
        mkdir(dir__)

        with open(join(dir__, 'description.json'), 'w') as fo:
            self.command['note'] = 'enter your note'
            json.dump(self.command, fo)

        with open(join(dir__, 'training_log.json'), 'w') as fo:
            json.dump(self.train_log, fo)

        with open(join(dir__, 'testing_log.json'), 'w') as fo:
            json.dump(self.test_log, fo)
