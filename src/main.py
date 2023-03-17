import argparse
import os
import torch

from data.mnist_loader import get_loader as m_loader
from server.base_server import BaseServer
from model.mnist_model import Mnist

MNIST_DATA = MNIST_MODEL = 'mnist'
CIFAR_DATA = CIFAR_MODEL = 'cifar'

FED_AVG = 'fed_avg'
FED_MAML = 'fed_maml'
FED_IMAML = 'fed_imaml'

def config_dataset(dataset:str):
    print('Preparing dataset')
    train_loaders = []
    test_support_loaders = []
    test_query_loaders = []

    if dataset == MNIST_DATA:
        train_dir = '../data/mnist/client_train'
        test_dir = '../data/mnist/client_test'
        loader = m_loader
    elif dataset == CIFAR_DATA:
        train_dir = '../data/cifar/client_train'
        test_dir = '../data/cifar/client_test'

    for train_file in os.listdir(train_dir):
        train_loaders.append(loader(os.path.join(train_dir, train_file)))

    list_test = os.listdir(test_dir)
    list_test.sort()
    for test_file in list_test:
        if 'q' in test_file:
            test_query_loaders.append(loader(os.path.join(test_dir, test_file)))
        else:
            test_support_loaders.append(loader(os.path.join(test_dir, test_file)))
    return train_loaders, test_support_loaders, test_query_loaders

def config_model(model:str):
    print('Preparing model')
    if model == MNIST_MODEL:
        return Mnist()
    elif model == CIFAR_MODEL:
        pass

def get_server(args: argparse.Namespace):
    print('Preparing server')
    train_loaders, test_support_loaders, test_query_loaders = config_dataset(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo = args.algorithm
    if algo == FED_AVG:
        return BaseServer(
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            device=device,
            local_lr=args.local_lr,
            global_lr=args.global_lr,
            model=config_model(args.model),
            num_training_clients=args.client_per_round,
            training_loaders=train_loaders,
            testing_loader=test_query_loaders)
    elif algo == FED_MAML:
        pass
    elif algo == FED_IMAML:
        pass
    else:
        print('meo')


def main():
    parser = argparse.ArgumentParser(description="FL + iMAML")

    parser.add_argument("--global_epochs", type=int, required=True, help="Global epochs")
    parser.add_argument("--global_lr", type=float, required=True, help="Global learning rate", default=1e-3)
    parser.add_argument("--local_epochs", type=int, required=True, help="Local epochs")
    parser.add_argument("--local_lr", type=float, required=True, help="Local learning rate", default=1e-3)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset", choices=[MNIST_DATA, CIFAR_DATA], default='mnist')
    parser.add_argument("--model", type=str, required=True, help="Model", choices=[MNIST_MODEL, CIFAR_MODEL], default='mnist')
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm", choices=[FED_AVG, FED_MAML, FED_IMAML], default='fed_avg')
    parser.add_argument("--client_per_round", type=int, required=True, help="Number of client evolving in training each round", default=5)

    args = parser.parse_args()

    server = get_server(args)
    server.train()

if __name__ == '__main__':
    main()
