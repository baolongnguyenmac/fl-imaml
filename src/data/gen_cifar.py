import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from os.path import join, isfile, isdir
import json
import shutil
np.random.seed(69)

from .gen_mnist import divide_data_for_clients, divide_num_sample_into_intervals, get_data_by_label, redivide_data

def gen_cifar(
        dir='../data/cifar',
        num_training_clients=50,
        num_testing_clients=30,
        num_labels=10,
        num_labels_per_client=2
    ):
    """generate data for non-iid scenario (no local client for testing)
    """
    print('\n=========== Generating data ===========')

    training_intervals = int(num_training_clients/num_labels*num_labels_per_client)
    testing_intervals = int(num_testing_clients/num_labels*num_labels_per_client)

    print('\nCreate training data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    X_train, y_train, X_test, y_test = redivide_data(dir, transform=transform, dataset='cifar')
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