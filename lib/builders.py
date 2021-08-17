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

from lib import networks


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
