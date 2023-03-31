from tqdm import tqdm
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.local_lr)
        num_sample = len(self.training_loader.dataset)

        for _ in range(self.local_epochs):
            training_loss, correct = 0., 0.
            for batch in self.training_loader:
                loss, correct_ = self._training_step(batch, self.model)

                correct += correct_
                training_loss += loss.item()

                # Back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.num_training_sample = num_sample

        tqdm.write(f'Base client {self.id}: Training loss = {training_loss:.7f}, Training acc = {correct/num_sample*100:.2f}%')

        return training_loss, correct/num_sample

    def test(self):
        num_sample = len(self.testing_loader.dataset)
        testing_loss, correct = 0., 0.

        with torch.no_grad():
            for batch in self.testing_loader:
                loss, correct_ = self._training_step(batch, self.model)
                testing_loss += loss.item()
                correct += correct_

        return testing_loss, correct/num_sample