# Ref: https://github.com/sshkhr/imaml

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
import higher
import torch.nn.functional as F

from .base_client import BaseClient

def apply_grad(model:torch.nn.Module, grad:list[torch.Tensor]):
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

class FediMAMLClient(BaseClient):
    def __init__(
            self,
            local_epochs: int,
            local_lr: float,
            global_lr: float,
            model: torch.nn.Module,
            device: torch.device,
            id: int,
            training_loader: DataLoader = None,
            test_support_loader: DataLoader = None,
            test_query_loader: DataLoader = None,
            lambda_: float = 100.,
            cg_step: int = 5
        ) -> None:
        super().__init__(local_epochs, local_lr, model, device, id)
        self.global_lr = global_lr
        self.training_loader: DataLoader = training_loader
        self.test_support_loader: DataLoader = test_support_loader
        self.test_query_loader: DataLoader = test_query_loader

        self.lambda_:float = lambda_
        self.cg_step:int = cg_step

    def test(self):
        inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)
        with higher.innerloop_ctx(self.model, inner_opt, self.device, copy_initial_weights=False) as (fmodel, diffopt):
            for batch in self.test_support_loader:
                support_loss, _ = self._training_step(batch, fmodel)
                diffopt.step(support_loss)

            outer_loss = 0.
            correct = 0.
            num_sample = len(self.test_query_loader.dataset)
            for batch in self.test_query_loader:
                query_loss, query_correct = self._training_step(batch, fmodel)

                correct += query_correct
                outer_loss += query_loss.item()

        return outer_loss, correct/num_sample

    @torch.no_grad()
    def conjugate_grad(self, inner_grad:torch.Tensor, outer_grad:torch.Tensor, params:list[torch.Tensor]):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - self.hv_prod(inner_grad, x, params)
        p = r.clone().detach()
        for i in range(self.cg_step):
            Ap = self.hv_prod(inner_grad, p, params)
            alpha = (r @ r)/(p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new)/(r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return self.vec_to_grad(x)

    def vec_to_grad(self, vec:torch.Tensor):
        pointer = 0
        res = []
        for param in self.model.parameters():
            num_param = param.numel()
            res.append(vec[pointer:pointer+num_param].view_as(param).data)
            pointer += num_param
        return res

    @torch.enable_grad()
    def hv_prod(self, inner_grad:torch.Tensor, x:torch.Tensor, params):
        hv = torch.autograd.grad(inner_grad, params, retain_graph=True, grad_outputs=x)
        hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrix
        return hv/self.lambda_ + x

    def loss_fn(self, pred:list[torch.Tensor], y:list[torch.Tensor], local_params:list[torch.Tensor], global_params:list[torch.Tensor]):
        return F.cross_entropy(pred, y) + self.lambda_/2 * sum([((lp - gp) ** 2).sum() for gp, lp in zip(global_params, local_params)])

    def _training_step(self, batch:list, model:torch.nn.Module):
        X, y = batch[0].to(self.device), batch[1].to(self.device)
        pred = model(X)
        loss = self.loss_fn(pred, y, list(model.parameters()), list(self.model.parameters()))
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        return loss, correct

    def _outer_loop(self):
        outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.global_lr)
        inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)

        with higher.innerloop_ctx(self.model, inner_opt, self.device, copy_initial_weights=False) as (fmodel, diffopt):
            num_batch = len(self.training_loader)
            for _ in range(self.local_epochs):
                for idx, batch in enumerate(self.training_loader):
                    if idx <= 0.2*num_batch:
                        support_loss, _ = self._training_step(batch, fmodel)
                        diffopt.step(support_loss)

            inner_loss = 0.
            for idx, batch in enumerate(self.training_loader):
                if idx <= 0.2*num_batch:
                    support_loss, _ = self._training_step(batch, fmodel)
                    inner_loss += support_loss

            outer_loss = 0.
            correct = 0.
            num_sample = 0
            for idx, batch in enumerate(self.training_loader):
                if idx > 0.2*num_batch or num_batch == 1:
                    query_loss, query_correct = self._training_step(batch, fmodel)

                    outer_loss += query_loss
                    correct += query_correct
                    num_sample += len(batch[0])

            params = list(fmodel.parameters())
            inner_grad = parameters_to_vector(torch.autograd.grad(inner_loss, params, create_graph=True))
            outer_grad = parameters_to_vector(torch.autograd.grad(outer_loss, params))
            implicit_grad = self.conjugate_grad(inner_grad, outer_grad, params)

        outer_opt.zero_grad()
        apply_grad(self.model, implicit_grad)
        outer_opt.step()

        self.num_training_sample = num_sample

        tqdm.write(f'iMAML client {self.id}: Training loss = {outer_loss.item():.7f}, Training acc = {correct/num_sample*100:.2f}%')

        return outer_loss.item(), correct/num_sample

    def train(self):
        return self._outer_loop()
