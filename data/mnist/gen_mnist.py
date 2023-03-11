import numpy as np
import pandas as pd
from typing import Tuple
import json

NUM_CLIENT = 50
NUM_LABEL_PER_CLIENT = 2
NUM_LABEL = 10
INTERVALS = int(NUM_CLIENT/NUM_LABEL*NUM_LABEL_PER_CLIENT)


def get_X_y(data_path) -> Tuple[np.ndarray, np.ndarray]:
    print('\nRead data')
    raw_data = (pd.read_csv(data_path)).to_numpy()

    print('\nNormalize data')
    X = raw_data[:, 1:]/255
    y = raw_data[:, 0]

    print(X.shape, y.shape)
    return X, y


def get_data_by_label(X: np.array, y: np.array) -> list[np.ndarray]:
    data_by_label = []
    for label in range(0, 10):
        data_by_label.append(X[y == label][:, :])
        np.random.shuffle(data_by_label[-1])
    return data_by_label


def divide_num_sample_into_intervals(data_by_label: list[np.array]) -> list[list[int]]:
    num_sample_in_label: list[int] = []

    for i in range(NUM_LABEL):
        total = len(data_by_label[i])
        tmp = []
        for j in range(INTERVALS-1):
            val = np.random.randint(total//(10 + 1), total//2)
            tmp.append(val)
            total -= val
        tmp.append(total)
        num_sample_in_label.append(tmp)

    print('\nNum sample of client in each label')
    [print(x) for x in num_sample_in_label]
    return num_sample_in_label


def divide_data_for_clients(data_by_label: list[np.ndarray], num_sample_in_label: list[list[int]], labels):
    # shuffle label
    np.random.shuffle(labels)

    all_user = {}
    flag1 = {labels[i]: 0 for i in labels}  # track a label
    flag2 = {labels[i]: 0 for i in labels}  # track interval in a label

    for i in range(NUM_CLIENT):
        all_user[i] = {}
        idx = [i % 10, i % 10 + 1 if i % 10 != 9 else 0]
        label_of_client = [labels[t] for t in idx]

        for label in label_of_client:
            tmp = flag1[label]
            tmp_ = tmp + num_sample_in_label[label][flag2[label]]
            all_user[i][int(label)] = data_by_label[label][tmp:tmp_].tolist()
            flag1[label] = tmp_
            flag2[label] += 1
    return all_user

def gen_mnist():
    """generate data for non-iid scenario (no local client for testing)
    """
    print('\n=========== Generating data ===========')

    print('\nCreate training data')
    X_train, y_train = get_X_y('./mnist_train.csv')
    data_train_by_label = get_data_by_label(X_train, y_train)
    num_train_sample_in_label = divide_num_sample_into_intervals(data_train_by_label)
    all_user_train = divide_data_for_clients(data_train_by_label, num_train_sample_in_label, list(range(10)))

    print('\nCreate testing data')
    X_test, y_test = get_X_y('./mnist_test.csv')
    data_test_by_label = get_data_by_label(X_test, y_test)
    num_test_sample_in_label = divide_num_sample_into_intervals(data_test_by_label)
    all_user_test = divide_data_for_clients(data_test_by_label, num_test_sample_in_label, list(range(10)))

    print('\n========')

    print('\nWrite data to file')
    for user in all_user_train.keys():
        with open(f'./client_train/{user}.json', 'w') as fp:
            json.dump(all_user_train[user], fp)

    for user in all_user_test.keys():
        with open(f'./client_test/{user}.json', 'w') as fp:
            json.dump(all_user_test[user], fp)

if __name__=='__main__':
    gen_mnist()
