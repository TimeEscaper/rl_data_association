import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from da_network import DANetwork


def main():
    n_observations = 2
    n_landmarks = 10
    obs_dim = 2

    da_nn = DANetwork(n_observations * obs_dim, 10, n_observations+1, n_landmarks)

    test_input = torch.randn(n_landmarks, n_observations * obs_dim)
    test_out = da_nn.forward(test_input)
    print(test_out.shape)


if __name__ == "__main__":
    main()
