import os
from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from smartlearner.interfaces import Model
from smartlearner.utils import sharedX
from smartlearner.utils import load_dict_from_json_file, save_dict_to_json_file
import smartlearner.initializers as initer


class RNN(Model):
    def __init__(self, input_size, hidden_size, output_size):
        self.graph_updates = OrderedDict()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W = sharedX(value=np.zeros((input_size, hidden_size)), name='W', borrow=True)
        self.bw = sharedX(value=np.zeros(hidden_size), name='bw', borrow=True)

        # Recurrence weights
        self.U = sharedX(value=np.zeros((hidden_size, hidden_size)), name='U', borrow=True)

        self.V = sharedX(value=np.zeros((hidden_size, output_size)), name='V', borrow=True)
        self.bv = sharedX(value=np.zeros(output_size), name='bv', borrow=True)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)
        weights_initializer(self.U)
        weights_initializer(self.V)

    @property
    def updates(self):
        return self.graph_updates

    @property
    def parameters(self):
        return [self.W, self.bw, self.U, self.V, self.bv]

    def get_output(self, X):
        def _fprop(Xi, last_h):
            h = T.nnet.sigmoid(T.dot(Xi, self.W) + T.dot(last_h, self.U) + self.bw)
            out = T.dot(h, self.V) + self.bv
            return h, out

        (Hs, preact_outputs), updates = theano.scan(fn=_fprop,
                                                    outputs_info=[T.zeros((X.shape[0], self.hidden_size)), None],
                                                    sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        outputs = T.transpose(preact_outputs, axes=(1, 0, 2))

        #outputs = T.nnet.softmax(preact_outputs[-1])
        #outputs = T.tanh(preact_outputs)
        return outputs

    def use(self, X):
        directions = self.get_output(X)
        return directions

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        hyperparameters = {'input_size': self.input_size,
                           'hidden_size': self.hidden_size,
                           'output_size': self.output_size}
        save_dict_to_json_file(pjoin(path, "meta.json"), {"name": self.__class__.__name__})
        save_dict_to_json_file(pjoin(path, "hyperparams.json"), hyperparameters)

        params = {param.name: param.get_value() for param in self.parameters}
        np.savez(pjoin(path, "params.npz"), **params)

    @classmethod
    def load(cls, path):
        meta = load_dict_from_json_file(pjoin(path, "meta.json"))
        assert meta['name'] == cls.__name__

        hyperparams = load_dict_from_json_file(pjoin(path, "hyperparams.json"))

        model = cls(**hyperparams)
        parameters = np.load(pjoin(path, "params.npz"))
        for param in model.parameters:
            param.set_value(parameters[param.name])

        return model
