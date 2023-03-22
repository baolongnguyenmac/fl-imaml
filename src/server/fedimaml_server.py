import torch
from torch.utils.data import DataLoader

from .base_server import BaseServer
from client.fedimaml_client import FediMAMLClient

class FediMAMLServer(BaseServer):
    def __init__(
            self,
            global_epochs: int,
            local_epochs: int,
            device: torch.device,
            global_lr: float,
            local_lr: float,
            model: torch.nn.Module,
            num_activated_clients: int,
            training_loaders:list[DataLoader],
            test_support_loaders: list[DataLoader],
            test_query_loaders: list[DataLoader],
            command: dict,
            lambda_: float,
            cg_step: int
        ) -> None:
        super().__init__(global_epochs, device, global_lr, model, num_activated_clients, command)

        for idx, loader in enumerate(training_loaders):
            tmp_client = FediMAMLClient(local_epochs, local_lr, global_lr, self.model, device, idx, loader, None, None, lambda_, cg_step)
            self.training_clients.append(tmp_client)

        for idx, (support_loader, query_loader) in enumerate(zip(test_support_loaders, test_query_loaders)):
            tmp_client = FediMAMLClient(local_epochs, local_lr, global_lr, self.model, device, idx, None, support_loader, query_loader, lambda_, cg_step)
            self.testing_clients.append(tmp_client)