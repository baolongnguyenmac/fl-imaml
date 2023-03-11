# About dataset

## MNIST

- `./mnist/mnist_train.csv` and `./mnist/mnist_test.csv` are downloaded from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv.

- `./mnist/mnist.csv` is constructed from these 2 above files.

- `./mnist/client_train` and `./mnist/client_test` contain data divided for clients using `./mnist/gen_mnist.py`. Specifically:
    - A non-iid data scenario is simulated. Each client contains data from only 2 labels and the labels between 2 clients are different.
    - Train: 80% MNIST dataset, Test: 80% MNIST dataset.

## CIFAR-10

- I'm gonna try using `./mnist/gen_mnist.py` to generate data for CIFAR-10 clients.
