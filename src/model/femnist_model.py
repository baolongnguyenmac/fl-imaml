import torch
import torch.nn as nn
import torch.nn.functional as F

class Femnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 62)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# from data.femnist_loader import get_loader

# model = Femnist()
# dataloader = get_loader('../data/femnist/client_test/0_s.json')
# for batch in dataloader:
#     print(batch[1])
#     print(model(batch[0]).size())
#     break
