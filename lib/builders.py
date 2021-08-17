#!/usr/bin/env python3
# Copyright 2019 Alexander Meulemans
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import torch

from lib import networks


def generate_data_from_teacher_network(teacher, n_in, num_train,
                                       num_test=100):
    """
    Generate a dataset by feeding random inputs through the given teacher
    network.
    Args:
        teacher: Teacher network for generating the dataset
        n_in: dimension of the inputs
        num_train: number of needed training samples
        num_test: number of needed test samples

    Returns:
        (tuple): Tuple containing:

        - **train_x**: Generated training inputs.
        - **test_x**: Generated test inputs.
        - **train_y**: Generated training outputs.
        - **test_y**: Generated test outputs.

        Data is returned in form of 2D arrays of class :class:`numpy.ndarray`.

    """
    ### Ensure deterministic computation.

    rand = np.random
    train_x = torch.rand(low=-1, high=1, size=(num_train, n_in))
    test_x = rand.uniform(low=-1, high=1, size=(num_test, n_in))

    train_y = teacher.forward(torch.from_numpy(train_x).float()).detach(). \
        numpy()
    test_y = teacher.forward(torch.from_numpy(test_x).float()).detach(). \
        numpy()

    return train_x, test_x, train_y, test_y


def generate_data_from_teacher(args, num_train=1000, num_test=100, n_in=5, n_out=5,
                               n_hidden=[10, 10, 10], activation='tanh',
                               device=None, num_val=None):
    """Generate data for a regression task through a teacher model.

    This function generates random input patterns and creates a random MLP
    (fully-connected neural network), that is used as a teacher model. I.e., the
    generated input data is fed through the teacher model to produce target
    outputs. The so produced dataset can be used to train and assess a
    student model. Hence, a learning procedure can be verified by validating its
    capability of training a student network to mimic a given teacher network.

    Input samples will be uniformly drawn from a unit cube.

    .. warning::
        Since this is a synthetic dataset that uses random number generators,
        the generated dataset depends on externally configured random seeds
        (and in case of GPU computation, it also depends on whether CUDA
        operations are performed in a derterministic mode).

    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        n_in (int): Passed as argument ``n_in`` to class
            :class:`lib.networks.DTPNetwork`
            when building the teacher model.
        n_out (int): Passed as argument ``n_out`` to class
            :class:`lib.networks.DTPNetwork`
            when building the teacher model.
        n_hidden (list): Passed as argument ``n_hidden`` to class
            :class:`lib.networks.DTPNetwork` when building the teacher model.
        activation (str): Passed as argument ``activation`` to
            class :class:`lib.networks.DTPNetwork` when building the
            teacher model

    Returns:
        See return values of function :func:`regression_cubic_poly`.
    """
    # FIXME Disentangle the random seeds set in a simulation from the one used
    # to generate synthetic datasets.
    if device is None:
        device = torch.device('cpu')
    if num_val is None:
        num_val = num_test
    # make sure that the same dataset is generated for each run
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    rand = np.random

    train_x = rand.uniform(low=-1, high=1, size=(num_train, n_in))
    test_x = rand.uniform(low=-1, high=1, size=(num_test, n_in))
    val_x = rand.uniform(low=-1, high=1, size=(num_val, n_in))
    # train_x = 0.01*rand.randn(num_train, n_in)
    # test_x = 0.01*rand.randn(num_test, n_in)

    # Note: make sure that gain is high, such that the neurons are pushed into
    # nonlinear regime. Otherwise we have a linear dataset
    teacher = networks.DTPNetwork(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                                  activation=activation, output_activation='linear',
                                  bias=True, initialization='teacher')

    if args.double_precision:
        train_y = teacher.forward(torch.from_numpy(train_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
    else:
        train_y = teacher.forward(torch.from_numpy(train_x).float().to(device)) \
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).float().to(device)) \
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).float().to(device)) \
            .detach().cpu().numpy()

    return train_x, test_x, val_x, train_y, test_y, val_y


def build_network(args):
    n_hidden = [args.size_hidden] * args.num_hidden

    if args.network_type == 'LeeDTP':
        net = networks.LeeDTPNetwork(
            n_in=args.size_input,
            n_hidden=n_hidden,
            n_out=args.size_output,
            activation=args.hidden_activation,
            sigma=args.sigma,
            forward_requires_grad=False,
        )
    elif args.network_type == 'DTP':
        net = networks.DTPNetwork(
            n_in=args.size_input,
            n_hidden=n_hidden,
            n_out=args.size_output,
            activation=args.hidden_activation,
            sigma=args.sigma,
            forward_requires_grad=False,
            fb_activation=args.fb_activation,
            plots=args.plots,
        )
    else:
        raise ValueError('The provided network type {} is not supported'.format(
            args.network_type
        ))
    return net
