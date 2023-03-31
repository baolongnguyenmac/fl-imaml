# About dataset

## MNIST

- `./mnist/client_train` and `./mnist/client_test` contain data divided for clients using `../src/data/gen_mnist.py`. Specifically:
    - A non-iid data scenario is simulated. Each client contains data from only 2 labels and the labels between 2 clients are different.
    - Train: 75% MNIST dataset, Test: 25% MNIST dataset.

## CIFAR-10

- Cifar-10 data is also generated in the same way.
