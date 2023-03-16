import torch
from copy import deepcopy
from torch.utils.data import DataLoader

class BaseClient:
    def __init__(
        self,
        local_epochs:int,
        local_lr:float,
        loader:DataLoader,
        model:torch.nn.Module,
        device:torch.device
    ) -> None:
        self.local_epochs:int = local_epochs
        self.local_lr:float = local_lr
        self.loader:DataLoader = loader
        self.model:torch.nn.Module = deepcopy(model)
        self.device = device

    def _set_param(self, model:torch.nn.Module):
        for local_p, global_p in zip(self.model.parameters(), model.parameters()):
            local_p.data = global_p.data.clone()

    def _train_step(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)

        size = len(self.loader.dataset)
        training_loss, correct = 0, 0

        for X, y in self.loader:
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred:torch.Tensor = self.model(X)
            loss:torch.Tensor = loss_fn(pred, y)
            training_loss += loss.item()

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        return training_loss/size, correct/size

    def test(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        size = len(self.loader.dataset)
        testing_loss, correct = 0, 0

        self.model.eval()
        with torch.no_grad():
            for X, y in self.loader:
                X, y = X.to(self.device), y.to(self.device)
                pred:torch.Tensor = self.model(X)
                testing_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        return testing_loss/size, correct/size

    def train(self):
        for _ in range(self.local_epochs):
            loss, acc = self._train_step()

        return loss, acc
