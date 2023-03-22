import torch
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from copy import deepcopy
import higher
import numpy as np
from src.model.mnist_model import Mnist
from src.data.mnist_loader import get_loader
from os.path import join, isfile, isdir
import os
from math import ceil
import torch.nn.functional as F

def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

def mix_grad(grad_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([g_list[i] for i in range(len(grad_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad

class iMAML:
    def __init__(
            self,
            global_epochs:int,
            local_epochs:int,
            global_lr:float,
            local_lr:float,
            model:torch.nn.Module,
            support_loaders:list[DataLoader],
            query_loaders:list[DataLoader],
            lambda_:float,
            n_cg:int
        ) -> None:

        self.global_epochs:int = global_epochs
        self.local_epochs:int = local_epochs
        self.local_lr:float = local_lr
        self.model:torch.nn.Module = deepcopy(model)
        self.support_loaders:list[DataLoader] = support_loaders
        self.query_loaders:list[DataLoader] = query_loaders
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=global_lr)
        self.lambda_:float = lambda_
        self.n_cg:int = n_cg

    def _inner_loop(self, model, batch, loss_fn):
        X, y = batch[0], batch[1]
        pred = model(X)
        loss = loss_fn(pred, y)
        return loss, (pred.argmax(1) == y).type(torch.float).sum().item()

    def loss_fn(self, pred:list[torch.Tensor], y:list[torch.Tensor], local_params:list[torch.Tensor], global_params:list[torch.Tensor]):
        return F.cross_entropy(pred, y) + self.lambda_/2 * sum([((gp - lp) ** 2).sum() for gp, lp in zip(global_params, local_params)])

    @torch.no_grad()
    def cg(self, in_grad, outer_grad, params):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(in_grad, x, params)
        p = r.clone().detach()
        for i in range(self.n_cg):
            Ap = self.hv_prod(in_grad, p, params)
            alpha = (r @ r)/(p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new)/(r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)

    def vec_to_grad(self, vec):
        pointer = 0
        res = []
        for param in self.model.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer+num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, in_grad, x, params):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrix
        return hv/self.lambda_ + x

    def _outer_loop(self, epoch:int, is_train:bool=True):
        tasks_per_round = 5
        count = 0
        num_batch_task = ceil(len(self.support_loaders)/tasks_per_round)
        loss_fn = torch.nn.CrossEntropyLoss()

        # lặp qua tất cả các batch task
        for batch_task_idx in range(num_batch_task):
            inner_loss = 0.
            outer_loss = 0.
            inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)

            accuracies = []
            grad_list = []

            # lặp qua tất cả các task trong batch_task
            for task_idx in range(count, count + tasks_per_round):
                if task_idx >= len(self.support_loaders):
                    break

                with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    for _ in range(self.local_epochs):
                        for batch in self.support_loaders[task_idx]:
                            support_loss, _ = self._inner_loop(fmodel, batch, loss_fn)
                            diffopt.step(support_loss)

                        for batch in self.support_loaders[task_idx]:
                            support_loss, _ = self._inner_loop(fmodel, batch, loss_fn)
                            inner_loss += support_loss

                    correct = 0.
                    for batch in self.query_loaders[task_idx]:
                        query_loss, correct = self._inner_loop(fmodel, batch, loss_fn)
                        outer_loss += query_loss

                    if is_train:
                        params = list(fmodel.parameters())
                        inner_grad = parameters_to_vector(torch.autograd.grad(inner_loss, params, create_graph=True))
                        outer_grad = parameters_to_vector(torch.autograd.grad(outer_loss, params))
                        implicit_grad = self.cg(inner_grad, outer_grad, params)
                        grad_list.append(implicit_grad)

                    # log info for this task
                    accuracies.append(correct/len(self.query_loaders[task_idx].dataset))
                    print(f'[Task {task_idx}]: Loss={query_loss.item():.5f}, Acc={accuracies[-1]*100:.2f}%')

            mean_acc, std = np.mean(accuracies)*100, np.std(accuracies)*100
            if is_train:
                self.outer_opt.zero_grad()
                grad = mix_grad(grad_list)
                apply_grad(self.model, grad)
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

    learner = iMAML(15, 2, 0.001, 0.001, Mnist(), support_loaders, query_loaders, 100., 5)
    learner.train()
