import torch
from torch.utils.data import DataLoader

from .base_client import BaseClient

class FedAvgClient(BaseClient):
    def __init__(
            self,
            local_epochs: int,
            local_lr: float,
            model: torch.nn.Module,
            device: torch.device,
            id: int,
            training_loader: DataLoader=None,
            testing_loader: DataLoader=None,
        ) -> None:
        super().__init__(local_epochs, local_lr, model, device, id)

        self.training_loader: DataLoader = training_loader
        self.testing_loader: DataLoader = testing_loader

    def train(self):
        print(f'Client {self.id}: Training base client')

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        optimizer.zero_grad()

        num_sample = len(self.training_loader.dataset)

        self.model.train()
        for _ in range(self.local_epochs):
            training_loss, correct = 0., 0.
            for batch in self.training_loader:
                loss, correct_ = self._training_step(batch, self.model, loss_fn)

                correct += correct_
                training_loss += loss.item()

                # Back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.num_training_sample = num_sample

        return training_loss, correct/num_sample

    def test(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        num_sample = len(self.testing_loader.dataset)
        testing_loss, correct = 0, 0

        self.model.eval()
        with torch.no_grad():
            for X, y in self.testing_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred:torch.Tensor = self.model(X)
                testing_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        return testing_loss, correct/num_sample