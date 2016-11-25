from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from learn2track.models.layers import LayerDense
from smartlearner import utils as smartutils
from smartlearner.interfaces import Model

floatX = theano.config.floatX


class FFNN(Model):
    """ A standard FFNN model with no output layer.

    The output is simply the state of the last hidden layer.
    """

    def __init__(self, input_size, hidden_sizes, activation='tanh'):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element X has.
        hidden_sizes : int, list of int
            Number of hidden units the model should have.
        activation : str
            Name of the activation function to use in the hidden layers
        """
        self.graph_updates = OrderedDict()
        self._gen = None

        self.input_size = input_size
        self.hidden_sizes = [hidden_sizes] if type(hidden_sizes) is int else hidden_sizes
        self.activation = activation

        self.layers = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers.append(LayerDense(last_hidden_size, hidden_size, activation=activation, name="Dense{}".format(i)))
            last_hidden_size = hidden_size

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for layer in self.layers:
            layer.initialize(weights_initializer)

    @property
    def updates(self):
        return self.graph_updates

    @property
    def hyperparameters(self):
        hyperparameters = {'version': 1,
                           'input_size': self.input_size,
                           'hidden_sizes': self.hidden_sizes,
                           'activation': self.activation}

        return hyperparameters

    @property
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters

        return parameters

    def get_init_states(self, batch_size):
        states_h = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            state_h = np.zeros((batch_size, hidden_size), dtype=floatX)
            states_h.append(state_h)

        return states_h

    def _fprop(self, Xi):
        layers_h = []

        next_input = Xi
        for i, layer in enumerate(self.layers):
            layer_output = layer.fprop(next_input)
            layers_h.append(layer_output)
            next_input = layer_output

        return tuple(layers_h)

    def get_output(self, X):
        last_layer_output = self._fprop(X)[-1]
        return last_layer_output

    def save(self, path):
        savedir = smartutils.create_folder(pjoin(path, type(self).__name__))
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), self.hyperparameters)

        params = {param.name: param.get_value() for param in self.parameters}
        assert len(self.parameters) == len(params)  # Implies names are all unique.
        np.savez(pjoin(savedir, "params.npz"), **params)

    def load(self, path):
        loaddir = pjoin(path, type(self).__name__)

        parameters = np.load(pjoin(loaddir, "params.npz"))
        for param in self.parameters:
            param.set_value(parameters[param.name])

    @classmethod
    def create(cls, path, **kwargs):
        loaddir = pjoin(path, cls.__name__)
        hyperparams = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        hyperparams.update(kwargs)

        model = cls(**hyperparams)
        model.load(path)
        return model
