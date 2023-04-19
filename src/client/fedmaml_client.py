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
            # training_loader: DataLoader = None,
            train_support_loader: DataLoader = None,
            train_query_loader: DataLoader = None,
            test_support_loader: DataLoader = None,
            test_query_loader: DataLoader = None,
        ) -> None:
        # in meta client: loader means training_loader
        super().__init__(local_epochs, local_lr, model, device, id)
        self.global_lr = global_lr
        # self.training_loader: DataLoader = training_loader
        self.train_support_loader: DataLoader = train_support_loader
        self.train_query_loader: DataLoader = train_query_loader
        self.test_support_loader: DataLoader = test_support_loader
        self.test_query_loader: DataLoader = test_query_loader

    def test(self):
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        with higher.innerloop_ctx(self.model, inner_opt, self.device, track_higher_grads=False) as (fmodel, diffopt):
            for _ in range(self.local_epochs):
                for batch in self.test_support_loader:
                    support_loss, _ = self._training_step(batch, fmodel)
                    diffopt.step(support_loss)

            with torch.no_grad():
                outer_loss = 0.
                correct = 0.
                for batch in self.test_query_loader:
                    query_loss, query_correct = self._training_step(batch, fmodel)

                    correct += query_correct
                    outer_loss += query_loss.item()

        return outer_loss, correct/len(self.test_query_loader.dataset)

    def _outer_loop(self):
        outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.global_lr)
        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)

        with higher.innerloop_ctx(self.model, inner_opt, self.device, copy_initial_weights=False, track_higher_grads=True) as (fmodel, diffopt):
            outer_opt.zero_grad()
            for _ in range(self.local_epochs):
                for batch in self.train_support_loader:
                    support_loss, _ = self._training_step(batch, fmodel)
                    diffopt.step(support_loss)

            outer_loss = 0.
            correct = 0.
            for batch in self.train_query_loader:
                query_loss, query_correct = self._training_step(batch, fmodel)

                # back propagation
                # query_loss.backward()

                outer_loss += query_loss
                correct += query_correct

            outer_loss /= len(self.train_query_loader)
            outer_loss.backward()

        outer_opt.step()

        self.num_training_sample = len(self.train_query_loader.dataset)
        tqdm.write(f'MAML client {self.id}: Training loss = {outer_loss.item():.7f}, Training acc = {correct/self.num_training_sample*100:.2f}%')
        return outer_loss.item(), correct/self.num_training_sample

    def train(self):
        return self._outer_loop()
