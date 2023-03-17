import torch
from torch.utils.data import DataLoader

from client.base_client import BaseClient
from .base_server import BaseServer

class FedAvgServer(BaseServer):
    def __init__(
            self,
            global_epochs: int,
            local_epochs:int,
            device: torch.device,
            global_lr: float,
            local_lr:float,
            model: torch.nn.Module,
            num_activated_clients: int,
            training_loaders:list[DataLoader],
            testing_loader:list[DataLoader],
            command: dict) -> None:
        super().__init__(global_epochs, device, global_lr, model, num_activated_clients, command)

        # init data for client
        for idx, loader in enumerate(training_loaders):
            tmp_client = BaseClient(local_epochs, local_lr, loader, model, device, idx)
            self.training_clients.append(tmp_client)

        for idx, loader in enumerate(testing_loader):
            tmp_client = BaseClient(local_epochs, local_lr, loader, model, device, idx)
            self.testing_clients.append(tmp_client)
