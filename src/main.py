import argparse
import os
import torch

from data.mnist_loader import get_loader as m_loader
from server.fedavg_server import FedAvgServer
from server.fedmaml_server import FedMAMLServer
from server.fedimaml_server import FediMAMLServer
from model.mnist_model import Mnist
from model.cifar_model import Cifar

MNIST_DATA = MNIST_MODEL = 'mnist'
CIFAR_DATA = CIFAR_MODEL = 'cifar'

FED_AVG = 'fed_avg'
FED_MAML = 'fed_maml'
FED_IMAML = 'fed_imaml'

def config_dataset(dataset:str):
    print('\nPreparing dataset ...')
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
    print('\nPreparing model ...')
    if model == MNIST_MODEL:
        return Mnist()
    elif model == CIFAR_MODEL:
        return Cifar()

def get_server(args: argparse.Namespace, command:dict):
    print('\nPreparing server ...')
    train_loaders, test_support_loaders, test_query_loaders = config_dataset(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\nUsing {device.type}')

    algo = args.algorithm
    if algo == FED_AVG:
        return FedAvgServer(
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            device=device,
            global_lr=args.global_lr,
            local_lr=args.local_lr,
            model=config_model(args.model).to(device),
            num_activated_clients=args.clients_per_round,
            training_loaders=train_loaders,
            testing_loaders=test_query_loaders,
            command=command)
    elif algo == FED_MAML:
        return FedMAMLServer(
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            device=device,
            global_lr=args.global_lr,
            local_lr=args.local_lr,
            model=config_model(args.model).to(device),
            num_activated_clients=args.clients_per_round,
            training_loaders=train_loaders,
            test_support_loaders=test_support_loaders,
            test_query_loaders=test_query_loaders,
            command=command)
    elif algo == FED_IMAML:
        return FediMAMLServer(
            global_epochs=args.global_epochs,
            local_epochs=args.local_epochs,
            device=device,
            global_lr=args.global_lr,
            local_lr=args.local_lr,
            model=config_model(args.model).to(device),
            num_activated_clients=args.clients_per_round,
            training_loaders=train_loaders,
            test_support_loaders=test_support_loaders,
            test_query_loaders=test_query_loaders,
            command=command,
            lambda_=args.lambda_,
            cg_step=args.cg_step)
    else:
        print('meo')


def main():
    parser = argparse.ArgumentParser(description="FL + iMAML")

    parser.add_argument("--global_epochs", type=int, required=True, help="Global epochs")
    parser.add_argument("--global_lr", type=float, required=False, help="Global learning rate")
    parser.add_argument("--local_epochs", type=int, required=True, help="Local epochs")
    parser.add_argument("--local_lr", type=float, required=True, help="Local learning rate")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset", choices=[MNIST_DATA, CIFAR_DATA], default='mnist')
    parser.add_argument("--model", type=str, required=True, help="Model", choices=[MNIST_MODEL, CIFAR_MODEL], default='mnist')
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm", choices=[FED_AVG, FED_MAML, FED_IMAML], default='fed_avg')
    parser.add_argument("--clients_per_round", type=int, required=True, help="Number of client evolving in training each round", default=5)
    parser.add_argument("--lambda_", type=float, required=False, help="Regularization hyper-param", default=100.)
    parser.add_argument("--cg_step", type=int, required=False, help="Conjugate step", default=5)

    args = parser.parse_args()
    command:dict = vars(args)

    server = get_server(args, command)
    server.train()

if __name__ == '__main__':
    main()
