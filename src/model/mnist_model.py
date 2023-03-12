from data.loader.mnist_loader import get_loader
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist(nn.Module):
    def __init__(self, input_dim=784, mid_dim=100, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

######## test model ########


# data
train_loader = get_loader('./data/mnist/mnist_train.csv')
test_loader = get_loader('./data/mnist/mnist_test.csv')

# device
device = torch.device('mps')
print(f'Using {device}')

# model
model = Mnist().to(device)
print(model)

# opt
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= size
    correct /= size
    return loss, correct

# test
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    return test_loss, correct

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
    print(f"Train: Accuracy: {(100*train_acc):>0.1f}%, Loss: {test_loss:>8f} \n")
    if t == 0 or (t+1) % 10 == 0:
        test_loss, test_acc = test(test_loader, model)
print("Done!")
