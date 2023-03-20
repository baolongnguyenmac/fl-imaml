import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import higher
import numpy as np
from src.model.mnist_model import Mnist
from src.data.mnist_loader import get_loader
from os.path import join, isfile, isdir
import os
from math import ceil
import torch.nn.functional as F

class MAML:
    def __init__(
            self,
            global_epochs:int,
            local_epochs:int,
            global_lr:float,
            local_lr:float,
            model:torch.nn.Module,
            support_loaders:list[DataLoader],
            query_loaders:list[DataLoader]
        ) -> None:

        self.global_epochs:int = global_epochs
        self.local_epochs:int = local_epochs
        self.local_lr:float = local_lr
        self.model:torch.nn.Module = deepcopy(model)
        self.support_loaders:list[DataLoader] = support_loaders
        self.query_loaders:list[DataLoader] = query_loaders
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=global_lr)

    def _inner_loop(self, model, batch, loss_fn):
        X, y = batch[0], batch[1]
        pred = model(X)
        loss = loss_fn(pred, y)
        return loss, (pred.argmax(1) == y).type(torch.float).sum().item()

    def _outer_loop(self, epoch:int, is_train:bool=True):
        self.model.train()
        tasks_per_round = 5
        count = 0
        num_batch_task = ceil(len(self.support_loaders)/tasks_per_round)
        loss_fn = torch.nn.CrossEntropyLoss()

        # lặp qua tất cả các batch task
        for batch_task_idx in range(num_batch_task):
            outer_loss = 0.
            self.outer_opt.zero_grad()
            inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)

            accuracies = []

            # lặp qua tất cả các task trong batch_task
            for task_idx in range(count, count + tasks_per_round):
                if task_idx >= len(self.support_loaders):
                    break

                with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    for _ in range(self.local_epochs):
                        for batch in self.support_loaders[task_idx]:
                            # support_pred = fmodel(X)
                            support_loss, _ = self._inner_loop(fmodel, batch, loss_fn)
                            diffopt.step(support_loss)

                    correct = 0.
                    for batch in self.query_loaders[task_idx]:
                        # query_pred = fmodel(X)
                        query_loss, correct = self._inner_loop(fmodel, batch, loss_fn)
                        outer_loss += query_loss
                        # correct = (query_pred.argmax(1) == y).type(torch.float).sum().item()

                    # log info for this task
                    accuracies.append(correct/len(self.query_loaders[task_idx].dataset))
                    print(f'[Task {task_idx}]: Loss={query_loss.item():.5f}, Acc={accuracies[-1]*100:.2f}%')

            mean_acc, std = np.mean(accuracies)*100, np.std(accuracies)*100
            if is_train:
                outer_loss.backward()
                self.outer_opt.step()
                print(f'\n[Epoch {epoch}]: Training loss = {outer_loss.item():0.5f}, Training acc = {mean_acc:.2f}±{std:.2f}%\n')
            else:
                print(f'\n[Epoch {epoch}]: Testing loss = {outer_loss.item():0.5f}, Testing acc = {mean_acc:.2f}±{std:.2f}%\n')

            count += tasks_per_round


    def train(self):
        for outer_it in range(self.global_epochs):
            print(f'\n======= Epoch {outer_it} =======\n')
            self._outer_loop(epoch=outer_it)

            if outer_it == 0 or (outer_it+1)%5 == 0:
                self.test(outer_it)

    def test(self, epoch:int):
        return self._outer_loop(epoch=epoch, is_train=False)

if __name__=='__main__':
    print('\nPrepare data\n')
    support_loaders = []
    query_loaders = []

    dir = './data/mnist/client_test'
    for filename in os.listdir(dir):
        loader = get_loader(join(dir, filename))
        if 'q' in filename:
            query_loaders.append(loader)
        else:
            support_loaders.append(loader)

    learner = MAML(15, 2, 0.001, 0.001, Mnist(), support_loaders, query_loaders)
    learner.train()
