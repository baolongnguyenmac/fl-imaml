import torch
from torch.utils.data import DataLoader

from .base_server import BaseServer

class FedMAMLServer(BaseServer):
    def __init__(self, global_epochs: int, local_epochs: int, dataset: str, device: torch.device, local_lr: float, global_lr: float, algorithm: torch.optim.Optimizer, model: torch.nn.Module, num_training_clients: int, training_loaders: list[DataLoader], testing_loader: list[DataLoader]) -> None:
        super().__init__(global_epochs, local_epochs, dataset, device, local_lr, global_lr, algorithm, model, num_training_clients, training_loaders, testing_loader)

        # # init training and testing client
        # self.training_clients:list[BaseClient] = []
        # self.testing_clients:list[BaseClient] = []
        # self.selected_clients:list[BaseClient] = []

        # # init data for client
        # for loader in training_loaders:
        #     tmp_client = BaseClient(local_epochs, local_lr, loader, self.model)
        #     self.training_clients.append(tmp_client)

        # for loader in testing_loader:
        #     tmp_client = BaseClient(local_epochs, local_lr, loader, self.model)
        #     self.testing_clients.append(tmp_client)