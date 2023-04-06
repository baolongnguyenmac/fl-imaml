# run this command first in leaf repo
# sh preprocess.sh -s niid --sf 0.125 -k 50 -t user --tf 0.8

import json
import numpy as np
import os
from os.path import isdir, join
import shutil

def get_data(file_path:str):
    fi = open(file_path, 'r')
    data = json.load(fi)
    return data

def data_by_label(user_data:dict[str, list], divide_sup_query=True):
    feature = np.array(user_data['x'])
    label = np.array(user_data['y'])

    support_set = {}
    query_set = {}
    data = {}
    for l in set(label):
        tmp = feature[label == l][:,:]
        if divide_sup_query:
            s_length = int(len(tmp)*0.2)+1 # query và support giao nhau 1 mẫu với mỗi label
            support_set[int(l)] = tmp[:s_length].tolist()
            query_set[int(l)] = tmp[s_length-1:].tolist()
        else:
            data[int(l)] = tmp.tolist()

    if divide_sup_query:
        return support_set, query_set
    else:
        return data

def gen_femnist(train_dir:str, test_dir:str, stored_dir:str):
    train_data = get_data(train_dir)
    test_data = get_data(test_dir)

    print('\n=========== Generate data ===========')
    # generate support test and query test for client
    test_support_list, test_query_list = [], []
    for user_data in test_data['user_data'].keys():
        support_set, query_set = data_by_label(test_data['user_data'][user_data])
        test_support_list.append(support_set)
        test_query_list.append(query_set)
    print(f'\nGenerated {len(test_support_list)} support set and {len(test_query_list)} query set')

    # generate training data for client
    train_list = []
    for user_id in train_data['user_data'].keys():
        train_list.append(data_by_label(train_data['user_data'][user_id], divide_sup_query=False))

    print('\n=========== Write data to file ===========')
    if isdir(join(stored_dir, 'client_test')):
        shutil.rmtree(join(stored_dir, 'client_test'))
    if isdir(join(stored_dir, 'client_train')):
        shutil.rmtree(join(stored_dir, 'client_train'))
    os.mkdir(join(stored_dir, 'client_test'))
    os.mkdir(join(stored_dir, 'client_train'))

    print(f"\nWriting to {join(stored_dir, 'client_test')}")
    for idx, (supp_user_data, query_user_data) in enumerate(zip(test_support_list, test_query_list)):
        with open(join(stored_dir, f'client_test/{idx}_s.json'), 'w') as fo:
            json.dump(supp_user_data, fo)
        with open(join(stored_dir, f'client_test/{idx}_q.json'), 'w') as fo:
            json.dump(query_user_data, fo)

    print(f"\nWriting to {join(stored_dir, 'client_train')}")
    for idx, user_data in enumerate(train_list):
        with open(join(stored_dir, f'client_train/{idx}.json'), 'w') as fo:
            json.dump(user_data, fo)

    print('\nDone')
