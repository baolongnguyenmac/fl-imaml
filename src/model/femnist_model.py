import torch.nn as nn
import torch.nn.functional as F

class Femnist(nn.Module):
    def __init__(self):
        super(Femnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024,2048)
        self.fc2 = nn.Linear(2048,62)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x),dim=1)


# from data.femnist_loader import get_loader

# model = Femnist()
# loss_fn = nn.NLLLoss()
# dataloader = get_loader('../data/femnist/client_test/0_s.json')
# for batch in dataloader:
#     pred = model(batch[0])
#     loss = loss_fn(pred, batch[1])
#     print(loss)
#     break
