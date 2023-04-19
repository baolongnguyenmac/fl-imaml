import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from typing import Tuple
import json
import matplotlib.pyplot as plt
import os
from os.path import join, isfile, isdir
import shutil
np.random.seed(69)


def redivide_data(dir:str, transform:transforms.Compose, dataset:str='mnist'):
    print('\nReDivide: 75% training data, 25% testing data')

    # download dataset
    if dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root=dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=dir, train=False, download=True, transform=transform)
    elif dataset == 'cifar':
        train_set = torchvision.datasets.CIFAR10(root=dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=dir, train=False, download=True, transform=transform)
    else:
        print('meo')
        return

    # concat
    if dataset == 'mnist':
        X = np.concatenate((train_set.data, test_set.data)).reshape(-1, 28*28)/255.
    elif dataset == 'cifar':
        X = np.concatenate((train_set.data, test_set.data)).reshape(-1, 32*32*3)/255.
    y = np.concatenate((train_set.targets, test_set.targets))

    # shuffle
    index_array = np.arange(len(y))
    np.random.shuffle(index_array)
    X = X[index_array]
    y = y[index_array]

    # redivide
    X_train, y_train = X[int(0.25*len(y)):], y[int(0.25*len(y)):]
    X_test, y_test = X[:int(0.25*len(y))], y[:int(0.25*len(y))]
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def get_data_by_label(X: np.array, y: np.array, num_label=10) -> list[np.ndarray]:
    data_by_label = []
    for label in range(num_label):
        data_by_label.append(X[y == label][:, :])
        # shuffle to have client with different data after generating
        np.random.shuffle(data_by_label[-1])
    return data_by_label


def divide_num_sample_into_intervals(
        data_by_label: list[np.array],
        intervals:int,
        num_labels:int
    ) -> list[list[int]]:

    num_sample_in_label: list[list[int]] = []

    for i in range(num_labels):
        total = len(data_by_label[i])
        tmp = []
        for j in range(intervals-1):
            val = np.random.randint(total//11, total//2)
            tmp.append(val)
            total -= val
        tmp.append(total)
        num_sample_in_label.append(tmp)

    print('\nNum sample of client in each label')
    [print(x) for x in num_sample_in_label]
    return num_sample_in_label


def divide_data_for_clients(
        data_by_label: list[np.ndarray],
        num_sample_in_label: list[list[int]],
        labels:list[int],
        num_clients:int,
        divide_sup_query: bool = True
    ):

    # # shuffle to have client with different data after generating
    np.random.shuffle(labels)

    all_user = {}
    flag1 = {labels[i]: 0 for i in labels}  # track a label
    flag2 = {labels[i]: 0 for i in labels}  # track interval in a label

    for i in range(num_clients):
        idx = [i % 10, i % 10 + 1 if i % 10 != 9 else 0]
        label_of_client = [labels[t] for t in idx]

        if divide_sup_query:
            # divide client dataset -> support + query
            all_user[f'{i}_s'] = {}
            all_user[f'{i}_q'] = {}
        else:
            # not divide support + query set
            all_user[i] = {}

        for label in label_of_client:
            start = flag1[label]
            num_sample = num_sample_in_label[label][flag2[label]]
            end = start + num_sample

            if divide_sup_query:
                all_user[f'{i}_s'][int(label)] = data_by_label[label][start:start+int(num_sample*0.2)].tolist()
                all_user[f'{i}_q'][int(label)] = data_by_label[label][start+int(num_sample*0.2):end].tolist()
            else:
                all_user[i][int(label)] = data_by_label[label][start:end].tolist()

            flag1[label] = end
            flag2[label] += 1
    return all_user


def vis_sample(data_by_label: list[np.ndarray]):
    # vis label 0->9
    list_09 = [label[0, :] for label in data_by_label]

    fig, axs = plt.subplots(2, 5, (10, 5))
    fig.suptitle('Vis label 0 -> 9')
    for (ax, img) in zip(axs.flat, enumerate(list_09)):
        ax.set_title(img[0])
        img = img[1].reshape(28, 28)
        ax.imshow(img, cmap='gray')

    plt.show()


def gen_mnist(
        dir='../data/mnist/',
        num_training_clients=50,
        num_testing_clients=30,
        num_labels=10,
        num_labels_per_client=2
    ):
    """generate data for non-iid scenario (no local client for testing)
    """
    print('\n=========== Generate data ===========')

    training_intervals = int(num_training_clients/num_labels*num_labels_per_client)
    testing_intervals = int(num_testing_clients/num_labels*num_labels_per_client)

    print('\nCreate training data')
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    X_train, y_train, X_test, y_test = redivide_data(dir, transform=transform, dataset='mnist')
    data_train_by_label = get_data_by_label(X_train, y_train)
    num_train_sample_in_label = divide_num_sample_into_intervals(data_train_by_label, training_intervals, num_labels)
    all_user_train = divide_data_for_clients(data_train_by_label, num_train_sample_in_label, list(range(10)), num_training_clients, False)

    print('\nCreate testing data')
    data_test_by_label = get_data_by_label(X_test, y_test)
    num_test_sample_in_label = divide_num_sample_into_intervals(data_test_by_label, testing_intervals, num_labels)
    all_user_test = divide_data_for_clients(data_test_by_label, num_test_sample_in_label, list(range(10)), num_testing_clients)

    print('\n=========== Write data to file ===========')

    if isdir(join(dir, 'client_test')):
        shutil.rmtree(join(dir, 'client_test'))
    if isdir(join(dir, 'client_train')):
        shutil.rmtree(join(dir, 'client_train'))
    os.mkdir(join(dir, 'client_test'))
    os.mkdir(join(dir, 'client_train'))

    for user in all_user_train.keys():
        with open(join(dir, f'client_train/{user}.json'), 'w') as fo:
            json.dump(all_user_train[user], fo)

    for user in all_user_test.keys():
        with open(join(dir, f'client_test/{user}.json'), 'w') as fo:
            json.dump(all_user_test[user], fo)

    print('\nDone')


# if __name__ == '__main__':
#     gen_mnist('../../data/mnist')
