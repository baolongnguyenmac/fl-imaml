import torch
from torch.utils.data import DataLoader

from client.fedavg_client import FedAvgClient
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
            testing_loaders:list[DataLoader],
            command: dict) -> None:
        super().__init__(global_epochs, device, global_lr, model, num_activated_clients, command)

        # init data for client
        for idx, loader in enumerate(training_loaders):
            tmp_client = FedAvgClient(local_epochs, local_lr, self.model, device, idx, loader, None)
            self.training_clients.append(tmp_client)

        for idx, loader in enumerate(testing_loaders):
            tmp_client = FedAvgClient(local_epochs, local_lr, self.model, device, idx, None, loader)
            self.testing_clients.append(tmp_client)
