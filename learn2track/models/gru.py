from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from smartlearner.interfaces import Model
from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer

from learn2track.models.layers import LayerGRU, LayerRegression, LayerDense


class GRU(Model):
    """ A standard GRU model with no output layer.

    See GRU_softmax or GRU_regression for implementations with an output layer.

    The output is simply the state of the last hidden layer.
    """
    def __init__(self, input_size, hidden_sizes):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each LSTM should have.
        """
        self.graph_updates = OrderedDict()
        self._gen = None

        self.input_size = input_size
        self.hidden_sizes = [hidden_sizes] if type(hidden_sizes) is int else hidden_sizes

        self.layers = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers.append(LayerGRU(last_hidden_size, hidden_size, name="GRU{}".format(i)))
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
                           'hidden_sizes': self.hidden_sizes}

        return hyperparameters

    @property
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters

        return parameters

    def _fprop(self, Xi, *args):
        layers_h = []

        input = Xi
        for i, layer in enumerate(self.layers):
            last_h = args[i]
            h = layer.fprop(input, last_h)
            layers_h.append(h)
            input = h

        return tuple(layers_h)

    def get_output(self, X):
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h,
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        self.h = T.transpose(results[0], axes=(1, 0, 2))
        return self.h

    def seq_squeeze(self, tokeep):
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(self.states_h[i].get_value()[tokeep])

    def seq_reset(self, batch_size=None):
        """ Start a new batch of sequences. """
        if self._gen is None:
            self.states_h = []
            for i, hidden_size in enumerate(self.hidden_sizes):
                self.states_h.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_h".format(i)))

        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(np.zeros((batch_size, hidden_size), dtype=theano.config.floatX))

    def seq_next(self, input):
        """ Returns the next element in every sequence of the batch. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h
            new_states = self._fprop(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]

            self._gen = theano.function([X], output, updates=updates)

        return self._gen(input)

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
    def create(cls, path):
        loaddir = pjoin(path, cls.__name__)
        hyperparams = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))

        model = cls(**hyperparams)
        model.load(path)
        return model
