import torch
from copy import deepcopy
import torch.nn as nn

class BaseClient:
    def __init__(
        self,
        local_epochs:int,
        local_lr:float,
        # loader:DataLoader,
        model:torch.nn.Module,
        device:torch.device,
        id:int
    ) -> None:
        self.local_epochs:int = local_epochs
        self.local_lr:float = local_lr
        # self.loader:DataLoader = loader
        self.model:torch.nn.Module = deepcopy(model)
        self.device = device
        self.id = id
        self.num_training_sample = 0

    def _set_param(self, model:torch.nn.Module):
        for local_p, global_p in zip(self.model.parameters(), model.parameters()):
            local_p.data = global_p.data.clone()

    def test(self):
        raise NotImplementedError('override test method in subclass')

    def _training_step(self, batch:list, model:nn.Module):
        loss_fn = nn.NLLLoss()
        X, y = batch[0].to(self.device), batch[1].to(self.device)
        pred = model(X)
        loss = loss_fn(pred, y)
        correct = (torch.max(pred, 1)[1] == y).sum().item()
        return loss, correct

    def train(self):
        raise NotImplementedError('override train method in subclass')
