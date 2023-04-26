import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import shutil

def get_X_y(dir:str='./data/mnist/'):
    if 'emnist' in dir: # must put before 'emnist' because 'mnist' in dir is right if dir is 'emnist'
        print('Get data of emnist\n')
        dataset_preparer = torchvision.datasets.EMNIST
        shape_ = (-1,28*28)
    elif 'mnist' in dir:
        print('Get data of mnist\n')
        dataset_preparer = torchvision.datasets.MNIST
        shape_ = (-1,28*28)
    elif 'cifar100' in dir: # must put before 'cifar10' because 'cifar10' in dir is right if dir is 'cifar100'
        print('Get data of cifar100\n')
        dataset_preparer = torchvision.datasets.CIFAR100
        shape_ = (-1,32*32*3)
    elif 'cifar10' in dir:
        print('Get data of cifar10\n')
        dataset_preparer = torchvision.datasets.CIFAR10
        shape_ = (-1,32*32*3)

    # load data
    if 'emnist' in dir:
        train_set = dataset_preparer(root=dir, train=True, split='byclass', download=True, transform=transforms.ToTensor())
        test_set = dataset_preparer(root=dir, train=False, split='byclass', download=True, transform=transforms.ToTensor())
    else:
        train_set = dataset_preparer(root=dir, train=True, download=True, transform=transforms.ToTensor())
        test_set = dataset_preparer(root=dir, train=False, download=True, transform=transforms.ToTensor())

    # concatenate
    X = np.concatenate((train_set.data, test_set.data)).reshape(shape_)
    y = np.concatenate((train_set.targets, test_set.targets))

    return X, y

def get_data_by_label(X: np.array, y: np.array, num_label=10) -> list[np.ndarray]:
    data_by_label = []
    for label in range(num_label):
        data_by_label.append(X[y == label])
        # shuffle to have client with different data after generating
        np.random.shuffle(data_by_label[-1])
    return data_by_label

def draw_dis(axis, client:dict[int, list], title:str=None, num_labels:int=10):
    if num_labels > 10:
        x = [str(key) for key in client.keys()]
        y = [len(client[key]) for key in client.keys()]
        axis.bar(x, y)
        if title:
            axis.set_title(title)
    else:
        profile = np.zeros((num_labels,2))
        profile[:,0] = range(num_labels)
        for key in client.keys():
            profile[int(key),1] = len(client[key])
        axis.bar(x=profile[:,0], height=profile[:,1])
        if title:
            axis.set_title(title)

def vis_dis_client(dir:str, dataset:str, k:int, num_labels:int):
    train_dir = os.path.join(dir, dataset, f'client_train_{k}')
    test_dir  = os.path.join(dir, dataset, f'client_test_{k}')

    fig, ax = plt.subplots(nrows=8, ncols=10, figsize=(15, 15), sharey=True)
    fig.subplots_adjust(hspace=0.5)
    list_file = os.listdir(train_dir)
    list_file.sort()
    for idx, file_ in enumerate(list_file):
        fi = open(os.path.join(train_dir, file_), 'r')
        data = json.load(fi)
        draw_dis(ax.flat[idx], data, file_, num_labels)
    fig.suptitle(f'train {dataset}: k={k}')
    fig.show()

    fig, ax = plt.subplots(nrows=4, ncols=10, figsize=(15, 7), sharey=True)
    fig.subplots_adjust(hspace=0.5)
    list_file = os.listdir(test_dir)
    list_file.sort()
    for idx, file_ in enumerate(list_file):
        fi = open(os.path.join(test_dir, file_), 'r')
        data = json.load(fi)
        draw_dis(ax.flat[idx], data, file_, num_labels)
    fig.suptitle(f'test {dataset}: k={k}')
    fig.show()

def write_client(dir:str, id:int, client:dict[int, list], split_supp_qry:bool=False):
    if split_supp_qry:
        supp_set, qry_set = {}, {}
        for label in client.keys():
            num_samples = len(client[label])
            supp_set[label] = client[label][:int(num_samples*0.2)]
            qry_set[label] = client[label][int(num_samples*0.2):]

        test_supp_dir = os.path.join(dir, f'{id}_s.json')
        fo = open(test_supp_dir, 'w')
        json.dump(supp_set, fo)
        test_qry_dir = os.path.join(dir, f'{id}_q.json')
        fo = open(test_qry_dir, 'w')
        json.dump(qry_set, fo)
    else:
        train_dir = os.path.join(dir, f'{id}.json')
        fo = open(train_dir, 'w')
        json.dump(client, fo)

def write_all_clients(all_clients:dict[int, dict], data_path:str, k:int, num_clients:int=100):
    train_data_path = os.path.join(data_path, f'client_train_{k}')
    test_data_path = os.path.join(data_path, f'client_test_{k}')
    if os.path.isdir(train_data_path):
        shutil.rmtree(train_data_path)
    if os.path.isdir(test_data_path):
        shutil.rmtree(test_data_path)
    os.mkdir(train_data_path)
    os.mkdir(test_data_path)

    print(f'Write data to {train_data_path} and {test_data_path}\n')
    for client in all_clients.keys():
        if client < num_clients*0.8:
            write_client(dir=train_data_path, id=client, client=all_clients[client])
        else:
            write_client(dir=test_data_path, id=client, client=all_clients[client], split_supp_qry=True)

# type = {'uniform', 'dirichlet'}
def distribution_based_split_client(data_by_label:list[np.array], num_clients:int=100, num_labels:int=10, type:str='dirichlet'):
    # compute number of samples for each label
    num_samples_per_label = []
    for data in data_by_label:
        num_samples_per_label.append(len(data))

    num_samples_in_client = []
    # compute the percentages data of each label in each client
    if type=='uniform':
        # split using Uniform distribution
        alpha = np.full(num_clients, 1/num_clients)
        for label in range(num_labels):
            num_samples_in_client.append((np.round(alpha * num_samples_per_label[label]).astype(int)).tolist())
    elif type=='dirichlet':
        # split using Distribution-based label imbalance mode
        alpha = np.full(num_clients, 0.5) # 0.5 from an A* paper
        for label in range(num_labels):
            p = np.random.dirichlet(alpha)
            num_samples_in_client.append((np.round(p * num_samples_per_label[label]).astype(int)).tolist())

    # split data to client
    all_clients = {}
    counts = {i:0 for i in range(num_labels)}
    for i in range(num_clients):
        all_clients[i] = {}
        for label in range(num_labels):
            num_samples = num_samples_in_client[label][i]
            if num_samples != 0:
                lower_bound = counts[label]
                upper_bound = lower_bound + num_samples
                all_clients[i][label] = data_by_label[label][lower_bound:upper_bound].tolist()
                counts[label] = upper_bound

    return all_clients

# Distribution-based label imbalance: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
def dirichlet_based_gen(dir='../../data'):
    num_clients = 100
    num_labels = [10, 10, 100, 62]
    k=4 # 4 for dirichlet based imbalance (alpha = 0.5)
    datasets = ['mnist', 'cifar10', 'cifar100', 'emnist']

    for idx, dataset in enumerate(datasets):
        data_path = os.path.join(dir, dataset)
        X, y = get_X_y(data_path)
        data_by_label = get_data_by_label(X, y, num_labels[idx])

        print(f'Generate {dataset}: Dirichlet distribution, num_clients={num_clients}, num_labels={num_labels[idx]}')
        all_clients = distribution_based_split_client(data_by_label, num_clients, num_labels[idx])
        write_all_clients(all_clients, data_path, k)

def iid_gen(dir='../../data'):
    num_clients = 100
    num_labels = [10, 10, 100, 62]
    k=5 # 5 for iid
    datasets = ['mnist', 'cifar10', 'cifar100', 'emnist']

    for idx, dataset in enumerate(datasets):
        data_path = os.path.join(dir, dataset)
        X, y = get_X_y(data_path)
        data_by_label = get_data_by_label(X, y, num_labels[idx])

        print(f'Generate {dataset}: IID, num_clients={num_clients}, num_labels={num_labels[idx]}')
        all_clients = distribution_based_split_client(data_by_label, num_clients, num_labels[idx], 'uniform')
        write_all_clients(all_clients, data_path, k)

# split using Quantity-based label imbalance mode
def quantity_based_split_client(data_by_label:list[np.array], k:int, num_clients:int=100, num_labels:int=10):
    partition_labels = {i:0 for i in range(num_labels)}

    # init all clients and partition labels
    all_clients = {}
    for i in range(num_clients):
        all_clients[i] = {}
        labels = np.random.choice(list(range(num_labels)), k, replace=False).tolist()
        for label in labels:
            all_clients[i][label] = None
            partition_labels[label] += 1

    # split sample of label into partitions
    labels_after_partition = {i:None for i in range(num_labels)}
    counts = {i:0 for i in range(num_labels)}
    for i in range(num_labels):
        if partition_labels[i] != 0:
            labels_after_partition[i] = np.array_split(data_by_label[i], partition_labels[i])

    # split data into clients
    for i in range(num_clients):
        for label in all_clients[i].keys():
            all_clients[i][label] = labels_after_partition[label][counts[label]].tolist()
            counts[label] += 1

    return all_clients

# Quantity-based label imbalance: each party owns data samples of a fixed number of labels.
def quantity_base_gen(dir='../../data'):
    ks = [1,2,3] # 1,2,3 for quantity-based imbalance
    num_clients = 100
    num_labels = [10, 10, 100, 62]
    datasets = ['mnist', 'cifar10', 'cifar100', 'emnist']

    for idx, dataset in enumerate(datasets):
        data_path = os.path.join(dir, dataset)
        X, y = get_X_y(data_path)
        data_by_label = get_data_by_label(X, y, num_labels[idx])

        for k in ks:
            print(f'Generate {dataset}: k={k}, num_clients={num_clients}, num_labels={num_labels[idx]}')
            all_clients = quantity_based_split_client(data_by_label, k, num_clients, num_labels[idx])

            # fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15, 15), sharey=True)
            # fig.subplots_adjust(hspace=0.3)
            # for client in all_clients.keys():
            #     draw_dis(ax.flat[client], all_clients[client], num_labels[idx])
            # fig.suptitle(f'Data {dataset}: k={k}, num_clients={num_clients}, num_labels={num_labels[idx]}')

            write_all_clients(all_clients, data_path, k)
