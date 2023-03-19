import torch
from torch.utils.data import DataLoader
from learn2learn.algorithms import MAML

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
        # create meta model
        meta_model: MAML = MAML(self.model, lr=self.local_lr, first_order=False)
        loss_fn = torch.nn.CrossEntropyLoss()

        # adapt
        learner:MAML = meta_model.clone()

        for _ in range(self.local_epochs):
            for idx, batch in enumerate(self.test_support_loader):
                support_loss, _ = self._training_step(batch, learner, loss_fn)
                learner.adapt(support_loss)

        testing_loss = 0.
        correct = 0.
        num_sample = len(self.test_query_loader.dataset)
        for idx, batch in enumerate(self.test_query_loader):
            query_loss, correct_ = self._training_step(batch, learner, loss_fn)

            correct += correct_
            testing_loss += query_loss.item()

        return testing_loss, correct/num_sample

    def train(self):
        print(f'Client {self.id}: Training MAML client')

        # create meta model
        meta_model: MAML = MAML(self.model, lr=self.local_lr, first_order=False)
        optimizer = torch.optim.SGD(meta_model.parameters(), self.global_lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        # run 1 round (#outer_loop=1)
        learner:MAML = meta_model.clone()

        num_batch = len(self.training_loader)
        for _ in range(self.local_epochs):
            for idx, batch in enumerate(self.training_loader):
                if idx <= 0.2*num_batch:
                    support_loss, _ = self._training_step(batch, learner, loss_fn)
                    learner.adapt(support_loss)

        training_loss = 0.
        correct = 0.
        num_sample = 0
        for idx, batch in enumerate(self.training_loader):
            if idx > 0.2*num_batch or num_batch == 1:
                query_loss, correct_ = self._training_step(batch, learner, loss_fn)

                num_sample += len(batch[0])
                correct += correct_
                training_loss += query_loss

        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        self.num_training_sample = num_sample

        return training_loss.item(), correct/num_sample


