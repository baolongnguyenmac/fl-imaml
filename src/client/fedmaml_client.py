# Ref: https://github.com/sshkhr/imaml

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import higher

from .base_client import BaseClient

class FedMAMLClient(BaseClient):
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
        ) -> None:
        # in meta client: loader means training_loader
        super().__init__(local_epochs, local_lr, model, device, id)
        self.global_lr = global_lr
        self.training_loader: DataLoader = training_loader
        self.test_support_loader: DataLoader = test_support_loader
        self.test_query_loader: DataLoader = test_query_loader

    def test(self):
        inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)
        with higher.innerloop_ctx(self.model, inner_opt, self.device, track_higher_grads=False) as (fmodel, diffopt):
            for _ in range(self.local_epochs):
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

    def _outer_loop(self):
        outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.global_lr)
        inner_opt = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)

        with higher.innerloop_ctx(self.model, inner_opt, self.device, track_higher_grads=True) as (fmodel, diffopt):
            num_batch = len(self.training_loader)
            for _ in range(self.local_epochs):
                for idx, batch in enumerate(self.training_loader):
                    if idx <= 0.2*num_batch:
                        support_loss, _ = self._training_step(batch, fmodel)
                        diffopt.step(support_loss)

            outer_loss = 0.
            correct = 0.
            num_sample = 0
            for idx, batch in enumerate(self.training_loader):
                if idx > 0.2*num_batch or num_batch == 1:
                    query_loss, query_correct = self._training_step(batch, fmodel)

                    outer_loss += query_loss
                    num_sample += len(batch[0])
                    correct += query_correct

        outer_opt.zero_grad()
        outer_loss.backward()
        outer_opt.step()

        self.num_training_sample = num_sample

        tqdm.write(f'MAML client {self.id}: Training loss = {outer_loss.item():.7f}, Training acc = {correct/num_sample*100:.2f}%')

        return outer_loss.item(), correct/num_sample

    def train(self):
        return self._outer_loop()
