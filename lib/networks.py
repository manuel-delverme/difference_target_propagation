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

import numpy as np
import torch
import torch.nn as nn

import lib.utils as utils
from lib.dtp_layers import DTPLayer


class DTPNetwork(nn.Module):
    """ A multilayer perceptron (MLP) network that will be trained by the
    difference target propagation (DTP) method.

    Attributes:
        layers (nn.ModuleList): a ModuleList with the layer objects of the MLP
        depth: the depth of the network (# hidden layers + 1)
        input (torch.Tensor): the input minibatch of the current training
                iteration. We need
                to save this tensor for computing the weight updates for the
                first hidden layer
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        update_idx (None or int): the layer index of which the layer parameters
            are updated for the current mini-batch, when working in a randomized
            setting. If the randomized setting is not used, it is equal to None.

    Args:
        n_in: input dimension (flattened input assumed)
        n_hidden: list with hidden layer dimensions
        n_out: output dimension
        activation: activation function indicator for the hidden layers
        output_activation: activation function indicator for the output layer
        bias: boolean indicating whether the network uses biases or not
        sigma: standard deviation of the gaussian that corrupts layer
                activations for computing the reconstruction losses.
        forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
        initialization (str): the initialization method used for the forward
                and feedback matrices of the layers


    """

    def __init__(self, n_in, n_hidden, n_out, activation='relu', output_activation='linear', bias=True, sigma=0.36, forward_requires_grad=False, initialization='orthogonal',
                 fb_activation='relu', plots=None):
        nn.Module.__init__(self)

        self._depth = len(n_hidden) + 1
        self._layers = self.set_layers(n_in, n_hidden, n_out, activation, output_activation, bias, forward_requires_grad, initialization, fb_activation)
        self._input = None
        self._sigma = sigma
        self._forward_requires_grad = forward_requires_grad
        self._update_idx = None
        self._plots = plots

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation, bias, forward_requires_grad, initialization, fb_activation):
        """
        Create the layers of the network and output them as a ModuleList.
        Args:
            n_in: input dimension (flattened input assumed)
            n_hidden: list with hidden layer dimensions
            n_out: output dimension
            activation: activation function indicator for the hidden layers
            output_activation: activation function indicator for the output
                layer
            bias: boolean indicating whether the network uses biases or not
            forward_requires_grad (bool): Flag indicating whether the forward
            parameters require gradients that can be computed with autograd.
            This might be needed when comparing the DTP updates with BP updates
            and GN updates.
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
            fb_activation (str): activation function indicator for the feedback
                path of the hidden layers

        """
        n_all = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            layers.append(
                DTPLayer(n_all[i - 1], n_all[i], bias=bias,
                         forward_activation=activation,
                         feedback_activation=fb_activation,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization
                         ))
        layers.append(DTPLayer(n_all[-2], n_all[-1], bias=bias, forward_activation=output_activation, feedback_activation=fb_activation,
                               forward_requires_grad=forward_requires_grad, initialization=initialization))
        return layers

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def layers(self):
        """Getter for read-only attribute :attr:`layers`."""
        return self._layers

    @property
    def sigma(self):
        """ Getter for read-only attribute sigma"""
        return self._sigma

    @property
    def input(self):
        """ Getter for attribute input."""
        return self._input

    @input.setter
    def input(self, value):
        """ Setter for attribute input."""
        self._input = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def update_idx(self):
        """ Getter for attribute update_idx"""
        return self._update_idx

    @update_idx.setter
    def update_idx(self, value):
        """Setter for attribute update_idx"""
        self._update_idx = value

    def forward(self, x):
        """ Propagate the input forward through the MLP network.

        Args:
            x: the input to the network

        returns:
            y: the output of the network
            """
        self.input = x
        y = x

        for layer in self.layers:
            y = layer.forward(y)

        # the output of the network requires a gradient in order to compute the
        # target (in compute_output_target() )
        if y.requires_grad == False:
            y.requires_grad = True

        return y

    def compute_output_target(self, loss, target_lr):
        """
        Compute the output target.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer

        Returns: Mini-batch of output targets
        """
        output_activations = self.layers[-1].activations

        gradient = torch.autograd.grad(loss, output_activations,
                                       retain_graph=self.forward_requires_grad) \
            [0].detach()
        output_targets = output_activations - \
                         target_lr * gradient
        return output_targets

    def propagate_backward(self, h_target, i):
        """
        Propagate the output target backwards to layer i in a DTP-like fashion.
        Args:
            h_target (torch.Tensor): the output target
            i: the layer index to which the target must be propagated

        Returns: the target for layer i

        """
        for k in range(self.depth - 1, i, -1):
            h_current = self.layers[k].activations
            h_previous = self.layers[k - 1].activations
            h_target = self.layers[k].backward(h_target, h_previous, h_current)
        return h_target

    def backward_random(self, loss, target_lr, i, save_target=False,
                        norm_ratio=1.):
        """ Propagate the output target backwards through the network until
        layer i. Based on this target, compute the gradient of the forward
        weights and bias of layer i and save them in the parameter tensors.
        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            i: layer index to which the target needs to be propagated and the
                gradients need to be computed
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
        """

        self.update_idx = i

        h_target = self.compute_output_target(loss, target_lr)

        h_target = self.propagate_backward(h_target, i)

        if save_target:
            self.layers[i].target = h_target

        if i == 0:  # first hidden layer needs to have the input
            # for computing gradients
            self.layers[i].compute_forward_gradients(h_target, self.input,
                                                     norm_ratio=norm_ratio)
        else:
            self.layers[i].compute_forward_gradients(h_target,
                                                     self.layers[i - 1].activations,
                                                     norm_ratio=norm_ratio)

    def backward_all(self, output_target, save_target=False, norm_ratio=1.):
        """ Propagate the output_target backwards through all the layers. Based
        on these local targets, compute the gradient of the forward weights and
        biases of all layers.

        Args:
            output_target (torch.Tensor): a mini-batch of targets for the
                output layer.
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
            """

        h_target = output_target

        if save_target:
            self.layers[-1].target = h_target
        for i in range(self.depth - 1, 0, -1):
            h_current = self.layers[i].activations
            h_previous = self.layers[i - 1].activations
            self.layers[i].compute_forward_gradients(h_target, h_previous,
                                                     norm_ratio=norm_ratio)
            h_target = self.layers[i].backward(h_target, h_previous, h_current)
            if save_target:
                self.layers[i - 1].target = h_target

        self.layers[0].compute_forward_gradients(h_target, self.input,
                                                 norm_ratio=norm_ratio)

    def backward(self, loss, target_lr, save_target=False, norm_ratio=1.):
        """ Compute and propagate the output_target backwards through all the
        layers. Based on these local targets, compute the gradient of the
        forward weights and biases of all layers.

        Args:
            loss (nn.Module): output loss of the network
            target_lr: the learning rate for computing the output target based
                on the gradient of the loss w.r.t. the output layer
            save_target (bool): flag indicating whether the target should be
                saved in the layer object for later use.
            norm_ratio (float): will only be used in children of DTPLayer for
                the minimal_norm update
            """

        output_target = self.compute_output_target(loss, target_lr)
        self.backward_all(output_target, save_target, norm_ratio=norm_ratio)

    def compute_feedback_gradients(self):
        """ Compute the local reconstruction loss for each layer and compute
        the gradient of those losses with respect to
        the feedback weights and biases. The gradients are saved in the
        feedback parameter tensors."""

        for i in range(1, self.depth):
            h_corrupted = self.layers[i - 1].activations + \
                          self.sigma * torch.randn_like(self.layers[i - 1].activations)
            self.layers[i].compute_feedback_gradients(h_corrupted, self.sigma)

    def get_forward_parameter_list(self):
        """
        Args:
            freeze_ouptut_layer (bool): flag indicating whether the forward
                parameters of the output layer should be excluded from the
                returned list.
        Returns: a list with all the forward parameters (weights and biases) of
            the network.

        """
        parameterlist = []
        for layer in self.layers:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_reduced_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters of the network, except
        from the ones of the output layer.
        """
        parameterlist = []
        for layer in self.layers[:-1]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_two_layers(self):
        parameterlist = []
        for layer in self.layers[-2:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_three_layers(self):
        parameterlist = []
        for layer in self.layers[-3:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameters_last_four_layers(self):
        parameterlist = []
        for layer in self.layers[-4:]:
            parameterlist.append(layer.weights)
            if layer.bias is not None:
                parameterlist.append(layer.bias)
        return parameterlist

    def get_forward_parameter_list_first_layer(self):
        """
        Returns: a list with only the forward parameters of the first layer.
        """
        parameterlist = []
        parameterlist.append(self.layers[0].weights)
        if self.layers[0].bias is not None:
            parameterlist.append(self.layers[0].bias)
        return parameterlist

    def get_feedback_parameter_list(self):
        """

        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.

        """
        parameterlist = []
        for layer in self.layers[1:]:
            parameterlist.append(layer.feedbackweights)
            if layer.feedbackbias is not None:
                parameterlist.append(layer.feedbackbias)
        return parameterlist

    def get_BP_updates(self, loss, i):
        """
        Compute the gradients of the loss with respect to the forward
        parameters of layer i.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index

        Returns (tuple): a tuple with the gradients of the loss with respect to
            the forward parameters of layer i, computed with backprop.

        """
        return self.layers[i].compute_bp_update(loss)

    def compute_bp_angles(self, loss, i, retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters.
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).

        """
        bp_gradients = self.layers[i].compute_bp_update(loss,
                                                        retain_graph)
        gradients = self.layers[i].get_forward_gradients()
        if utils.contains_nan(bp_gradients[0].detach()):
            print('bp update contains nan (layer {}):'.format(i))
            print(bp_gradients[0].detach())
        if utils.contains_nan(gradients[0].detach()):
            print('weight update contains nan (layer {}):'.format(i))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') < 1e-14:
            print('norm updates approximately zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())
        if torch.norm(gradients[0].detach(), p='fro') == 0:
            print('norm updates exactly zero (layer {}):'.format(i))
            print(torch.norm(gradients[0].detach(), p='fro'))
            print(gradients[0].detach())

        weights_angle = utils.compute_angle(bp_gradients[0].detach(),
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(bp_gradients[1].detach(),
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle,)

    def compute_gn_angles(self, output_activation, loss, damping, i,
                          retain_graph=False):
        """
        Compute the angles of the current forward parameter updates of layer i
        with the GN update for those parameters.
        Args:
            see lib.dtp_layers.DTPLayer.compute_gn_updates(...)
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        Returns (tuple): Tuple containing the angle in degrees between the
            updates for the forward weights at index 0 and the forward bias
            at index 1 (if bias is not None).

        """
        gn_gradients = self.layers[i].compute_gn_update(output_activation,
                                                        loss,
                                                        damping,
                                                        retain_graph)
        gradients = self.layers[i].get_forward_gradients()
        weights_angle = utils.compute_angle(gn_gradients[0],
                                            gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(gn_gradients[1],
                                             gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle,)

    def compute_gn_activation_angle(self, output_activation, loss,
                                    damping, i, step,
                                    retain_graph=False,
                                    linear=False):
        """
        Compute the angle between the difference between the target and layer
        activation and the gauss-newton update for the layers activation
        Args:
            see lib.dtp_layers.DTPLayer.compute_gn_activation_updates(...)
            i (int): layer index
            step (int): epoch step, just for logging
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.

        Returns: The average angle in degrees

        """
        if linear:
            target_difference = self.layers[i].linearactivations - \
                                self.layers[i].target
        else:
            target_difference = self.layers[i].activations - \
                                self.layers[i].target
        gn_updates = self.layers[i].compute_gn_activation_updates(
            output_activation,
            loss,
            damping,
            retain_graph=retain_graph,
            linear=linear
        )
        # print(f"Layer {i}:")
        # print(torch.mean(target_difference).item())
        # print(torch.mean(gn_updates).item())
        if self._plots is not None:
            self.td_activation.at[step, i] = torch.mean(target_difference).item()
            self.gn_activation.at[step, i] = torch.mean(gn_updates).item()

        # exit()
        gn_activationav = utils.compute_average_batch_angle(target_difference, gn_updates)
        return gn_activationav

    def compute_bp_activation_angle(self, loss, i, retain_graph=False,
                                    linear=False):
        """
        Compute the angle between the difference between the target and layer
        activation and the backpropagation update for the layers activation
        Args:
            loss (torch.Tensor): the loss value of the current minibatch.
            i (int): layer index
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
            linear (bool): Flag indicating whether the GN update for the
                linear activations should be computed instead of for the
                nonlinear activations.

        Returns : The average angle in degrees
        """
        if linear:
            target_difference = self.layers[i].linearactivations - \
                                self.layers[i].target
        else:
            target_difference = self.layers[i].activations - \
                                self.layers[i].target
        bp_updates = self.layers[i].compute_bp_activation_updates(
            loss=loss,
            retain_graph=retain_graph,
            linear=linear
        ).detach()

        angle = utils.compute_average_batch_angle(target_difference.detach(),
                                                  bp_updates)

        return angle

    def compute_gnt_angle(self, output_activation, loss, damping,
                          i, step, retain_graph=False, linear=False):
        if i == 0:
            h_previous = self.input
        else:
            h_previous = self.layers[i - 1].activations

        gnt_updates = self.layers[i].compute_gnt_updates(
            output_activation=output_activation,
            loss=loss,
            h_previous=h_previous,
            damping=damping,
            retain_graph=retain_graph,
            linear=linear
        )

        gradients = self.layers[i].get_forward_gradients()
        weights_angle = utils.compute_angle(gnt_updates[0], gradients[0])
        if self.layers[i].bias is not None:
            bias_angle = utils.compute_angle(gnt_updates[1], gradients[1])
            return (weights_angle, bias_angle)
        else:
            return (weights_angle,)

    def save_feedback_batch_logs(self, writer, step, init=False):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """
        for i in range(len(self.layers)):
            name = 'layer {}'.format(i + 1)
            self.layers[i].save_feedback_batch_logs(writer, step, name,
                                                    no_gradient=i == 0, init=init)

    def save_bp_angles(self, writer, step, loss, retain_graph=False):
        """
        Save the angles of the current forward parameter updates
        with the backprop update for those parameters on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            loss (torch.Tensor): the loss value of the current minibatch.
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_bp_angles(loss, i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_bp_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.bp_angles.at[step, i] = angles[0].item()

            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_bp_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def save_gn_angles(self, writer, step, output_activation, loss, damping,
                       retain_graph=False):
        """
        Save the angles of the current forward parameter updates of layer i
        with the backprop update for those parameters. on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_gn_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.

        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gn_angles(output_activation, loss, damping,
                                            i, retain_graph_flag)
            writer.add_scalar(
                tag='{}/weight_gn_angle'.format(name),
                scalar_value=angles[0],
                global_step=step
            )

            if self._plots is not None:
                self.gn_angles.at[step, i] = angles[0].item()

            if self.layers[i].bias is not None:
                writer.add_scalar(
                    tag='{}/bias_gn_angle'.format(name),
                    scalar_value=angles[1],
                    global_step=step
                )

    def save_gnt_angles(self, writer, step, output_activation, loss,
                        damping, retain_graph=False, custom_result_df=None):
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        # print('saving gnt angles')
        if self.update_idx is None:
            layer_indices = range(len(self.layers) - 1)
        else:
            layer_indices = [self.update_idx]

        # assign a damping constant for each layer for computing the gnt angles
        if isinstance(damping, float):
            damping = [damping for i in range(self.depth)]
        else:
            # print(damping)
            # print(len(damping))
            # print(layer_indices)
            # print(len(layer_indices))
            assert len(damping) == len(layer_indices)

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angles = self.compute_gnt_angle(output_activation=output_activation,
                                            loss=loss,
                                            damping=damping[i],
                                            i=i,
                                            step=step,
                                            retain_graph=retain_graph_flag)
            if custom_result_df is not None:
                custom_result_df.at[step, i] = angles[0].item()
            else:
                writer.add_scalar(
                    tag='{}/weight_gnt_angle'.format(name),
                    scalar_value=angles[0],
                    global_step=step
                )

                if self._plots is not None:
                    # print('saving gnt angles')
                    # print(angles[0].item())
                    self.gnt_angles.at[step, i] = angles[0].item()

                if self.layers[i].bias is not None:
                    writer.add_scalar(
                        tag='{}/bias_gnt_angle'.format(name),
                        scalar_value=angles[1],
                        global_step=step
                    )

    def save_nullspace_norm_ratio(self, writer, step, output_activation,
                                  retain_graph=False):
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph

            relative_norm = self.layers[i].compute_nullspace_relative_norm(
                output_activation,
                retain_graph=retain_graph_flag
            )

            writer.add_scalar(
                tag='{}/nullspace_relative_norm'.format(name),
                scalar_value=relative_norm,
                global_step=step
            )

            if self._plots is not None:
                self.nullspace_relative_norm.at[step, i] = relative_norm.item()

    def save_bp_activation_angle(self, writer, step, loss,
                                 retain_graph=False):
        """
        Save the angle between the difference between the target and layer
        activation and the backpropagation update for the layers activation
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_bp_activation_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angle = self.compute_bp_activation_angle(loss, i,
                                                     retain_graph_flag)

            writer.add_scalar(
                tag='{}/activation_bp_angle'.format(name),
                scalar_value=angle,
                global_step=step
            )
            if self._plots is not None:
                self.bp_activation_angles.at[step, i] = angle.item()
        return

    def save_gn_activation_angle(self, writer, step, output_activation, loss,
                                 damping, retain_graph=False):
        """
        Save the angle between the difference between the target and layer
        activation and the gauss-newton update for the layers activation
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            see lib.dtp_layers.DTPLayer.compute_bp_activation_updates(...)
            retain_graph (bool): flag indicating whether the graph of the
                network should be retained after computing the gradients or
                jacobians. If the graph will not be used anymore for the current
                minibatch afterwards, retain_graph should be False.
        """
        # if self.update_idx is None, the randomized setting is not used and
        # all the layers have their parameters updated. The angle should thus
        # be computed for all layers
        if self.update_idx is None:
            layer_indices = range(len(self.layers))
        else:
            layer_indices = [self.update_idx]

        for i in layer_indices:
            name = 'layer {}'.format(i + 1)
            if i != layer_indices[-1]:  # if it is not the last index, the graph
                # should be saved for the next index
                retain_graph_flag = True
            else:
                retain_graph_flag = retain_graph
            angle = self.compute_gn_activation_angle(output_activation, loss,
                                                     damping, i, step,
                                                     retain_graph_flag)
            writer.add_scalar(
                tag='{}/activation_gn_angle'.format(name),
                scalar_value=angle,
                global_step=step
            )

            if self._plots is not None:
                self.gn_activation_angles.at[step, i] = angle.item()
        return

    def get_av_reconstruction_loss(self):
        """ Get the average reconstruction loss of the network across its layers
        for the current mini-batch.
        Args:
            network (networks.DTPNetwork): network
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        reconstruction_losses = np.array([])

        for layer in self.layers[1:]:
            reconstruction_losses = np.append(reconstruction_losses,
                                              layer.reconstruction_loss)

        return np.mean(reconstruction_losses)


class LeeDTPNetwork(nn.Module):
    """ Class for the DTP network used in Lee2015 to classify MNIST digits. In
    this network, the target for the last hidden layer is computed by error
    backpropagation instead of DTP. """

    def __init__(self, n_in, n_hidden, n_out, activation='tanh', sigma=0.36):
        """ See arguments of __init__ of class DTP Network

        Attributes:
            dtpnetwork (DTPNetwork): a DTP Network of all the layers except
                from the output
                layer. These layers will be trained by the DTP method.
            linearlayer (nn.Linear): the output linear layer. On this layer, the
                CrossEntropyLoss will be applied during training.
            hiddengradient: the gradient of the loss with respect to the
                activation of the last hidden layer of the network.
            depth (int): depth of the network (number of hidden layers + 1)
            initialization (str): the initialization method used for the forward
                and feedback matrices of the layers
        """
        nn.Module.__init__(self)

        self._dtpnetwork = DTPNetwork(n_in, n_hidden[:-1], n_hidden[-1], activation=activation, output_activation=activation, sigma=sigma)
        self._linearlayer = nn.Linear(n_hidden[-1], n_out)

        nn.init.constant_(self._linearlayer.bias, 0)

        self._depth = len(n_hidden) + 1
        self._update_idx = None

    @property
    def dtpnetwork(self):
        """ Getter for read-only attribute dtpnetwork"""
        return self._dtpnetwork

    @property
    def linearlayer(self):
        """ Getter for read-only attribute linearlayer"""
        return self._linearlayer

    @property
    def depth(self):
        """Getter for read-only attribute :attr:`depth`."""
        return self._depth

    @property
    def update_idx(self):
        """ Getter for attribute update_idx"""
        return self._update_idx

    @update_idx.setter
    def update_idx(self, value):
        """Setter for attribute update_idx"""
        self._update_idx = value

    @property
    def forward_requires_grad(self):
        """ Getter for read-only attribute forward_requires_grad"""
        return self._forward_requires_grad

    @property
    def layers(self):
        """ Getter for attribute layers.
        Warning: only the layers of the dtp network are returned, not the
        extra linear layer output layer"""
        return self.dtpnetwork.layers

    def forward(self, x):
        x = self.dtpnetwork.forward(x)
        if not x.requires_grad:  # if statement is needed to be sure that
            # x is a leaf node. Otherwise, we are not
            # allowed to change the grad attribute.
            x.requires_grad = True
        x = self.linearlayer(x)

        return x

    def backward(self, loss, target_lr, save_target=False):
        """ Compute the gradients of the output weights and bias, compute
        the target for the last hidden layer based on backprop, propagate target
        backwards and compute parameter updates for the rest of the DTP network.
        """

        # compute the gradients of the weights and bias of the output linear
        # layer. We cannot do this with loss.backward(), as then the
        # gradients of all leaf nodes will be computed and stored in the .grad
        # attributes of all layer parameters with requires_grad=True. We only
        # need the gradients with respect to the last hidden layer and the
        # weight and bias of the output linear layer.

        gradients = torch.autograd.grad(loss, self.linearlayer.parameters(),
                                        retain_graph=True)
        for i, param in enumerate(self.linearlayer.parameters()):
            param.grad = gradients[i].detach()

        hidden_activations = self.dtpnetwork.layers[-1].activations
        hiddengradient = torch.autograd.grad(loss, hidden_activations,
                                             retain_graph=
                                             self.forward_requires_grad)
        hiddengradient = hiddengradient[0].detach()

        hidden_targets = hidden_activations - target_lr * hiddengradient
        self.dtpnetwork.backward_all(hidden_targets, save_target)

    def compute_feedback_gradients(self):
        """ Compute the local reconstruction loss for each layer of the
        dtp network and compute the gradient of those losses with respect to
        the feedback weights and biases. The gradients are saved in the
        feedback parameter tensors."""

        self.dtpnetwork.compute_feedback_gradients()

    def get_forward_parameter_list(self):
        """
        Returns: a list with all the forward parameters (weights and biases) of
            the network.

        """
        parameterlist = self.dtpnetwork.get_forward_parameter_list()
        parameterlist.append(self.linearlayer.weight)
        if self.linearlayer.bias is not None:
            parameterlist.append(self.linearlayer.bias)
        return parameterlist

    def get_feedback_parameter_list(self):
        """

        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.

        """
        return self.dtpnetwork.get_feedback_parameter_list()

    def get_reduced_forward_parameter_list(self):
        """
        Get the forward parameters of all the layers that will be trained by
        DTP, and not by BP (thus all the layer parameters except from the output
        layer and the last hidden layer.
        Returns: a list with all the parameters that will be trained by
            difference target propagtion

        """
        if self.dtpnetwork.layers[-1].bias is not None:
            return self.dtpnetwork.get_forward_parameter_list()[:-2]
        else:
            return self.dtpnetwork.get_forward_parameter_list()[:-1]

    def save_feedback_batch_logs(self, writer, step):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
        """
        self.dtpnetwork.save_feedback_batch_logs(writer, step)

    def save_bp_angles(self, writer, step, loss, retain_graph=False):
        """
        See DTPNetwork.save_bp_angles

        """
        self.dtpnetwork.save_bp_angles(writer, step, loss, retain_graph)

    def save_gn_angles(self, writer, step, output_activation, loss, damping,
                       retain_graph=False):
        """
        See DTPNetwork.save_gn_angles

        """
        self.dtpnetwork.save_gn_angles(writer, step, output_activation, loss,
                                       damping, retain_graph)

    def save_bp_activation_angle(self, writer, step, loss,
                                 retain_graph=False):
        """ See DTPNetwork.save_bp_activation_angle. """
        self.dtpnetwork.save_bp_activation_angle(writer, step, loss,
                                                 retain_graph)

    def save_gn_activation_angle(self, writer, step, output_activation, loss,
                                 damping, retain_graph=False):
        """ See DTPNetwork.save_gn_activation_angle. """

        self.dtpnetwork.save_gn_activation_angle(writer, step,
                                                 output_activation, loss,
                                                 damping, retain_graph)

    def get_av_reconstruction_loss(self):
        """ Get the average reconstruction loss of the network across its layers
        for the current mini-batch.
        Args:
            network (networks.DTPNetwork): network
        Returns (torch.Tensor):
            Tensor containing a scalar of the average reconstruction loss
        """
        return self.dtpnetwork.get_av_reconstruction_loss()
