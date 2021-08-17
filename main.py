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
# Modifications copyright (C) 2021 Manuel Del Verme
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import config
from lib import builders
from lib.train import train_bp


def run():
    experiment_buddy.register_defaults(vars(config))
    writer = experiment_buddy.deploy()

    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    print(config.device)
    print('Using cuda: ' + str(config.use_cuda))

    test_loader, train_loader, val_loader = load_dataset()

    if config.log_interval is None:
        config.log_interval = max(1, int(len(train_loader) / 100))

    net = builders.build_network(config).to(config.device)

    writer.watch(net, log="all")
    train_bp(args=config, device=config.device, train_loader=train_loader, net=net, writer=writer, test_loader=test_loader, val_loader=val_loader)

    if writer is not None:
        writer.close()


def load_dataset():
    if config.dataset == 'mnist':

        data_dir = './data'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        trainset_total = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

        trainset, valset = torch.utils.data.random_split(trainset_total, [55000, 5000])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    elif config.dataset == 'cifar10':
        data_dir = './data'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset_total = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        trainset, valset = torch.utils.data.random_split(trainset_total, [45000, 5000])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(config.dataset))
    return test_loader, train_loader, val_loader


if __name__ == '__main__':
    import experiment_buddy

    run()
