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


class LSTM(Model):
    def __init__(self, input_size, hidden_size, output_size):
        self.graph_updates = OrderedDict()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Input weights (i:input, o:output, f:forget, m:memory)
        self.Wi = sharedX(value=np.zeros((input_size, hidden_size)), name='Wi')
        self.Wo = sharedX(value=np.zeros((input_size, hidden_size)), name='Wo')
        self.Wf = sharedX(value=np.zeros((input_size, hidden_size)), name='Wf')
        self.Wm = sharedX(value=np.zeros((input_size, hidden_size)), name='Wm')

        # Biases (i:input, o:output, f:forget, m:memory)
        self.bi = sharedX(value=np.zeros(hidden_size), name='bi')
        self.bo = sharedX(value=np.zeros(hidden_size), name='bo')
        self.bf = sharedX(value=np.zeros(hidden_size), name='bf')
        self.bm = sharedX(value=np.zeros(hidden_size), name='bm')

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        self.Ui = sharedX(value=np.zeros((hidden_size, hidden_size)), name='Ui')
        self.Uo = sharedX(value=np.zeros((hidden_size, hidden_size)), name='Uo')
        self.Uf = sharedX(value=np.zeros((hidden_size, hidden_size)), name='Uf')
        self.Um = sharedX(value=np.zeros((hidden_size, hidden_size)), name='Um')

        # Memory weights (i:input, o:output, f:forget, m:memory)
        self.Vi = sharedX(value=np.eye(hidden_size), name='Vi')
        self.Vo = sharedX(value=np.eye(hidden_size), name='Vo')
        self.Vf = sharedX(value=np.eye(hidden_size), name='Vf')

        # Output weights and biases
        self.Z = sharedX(value=np.zeros((hidden_size, output_size)), name='Z')
        self.bz = sharedX(value=np.zeros(output_size), name='bz')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for param in [self.Wi, self.Wo, self.Wf, self.Wm]:
            weights_initializer(param)

        for param in [self.Ui, self.Uo, self.Uf, self.Um]:
            weights_initializer(param)

        weights_initializer(self.Z)

    @property
    def updates(self):
        return self.graph_updates

    @property
    def parameters(self):
        return [self.Wi, self.Wo, self.Wf, self.Wm,
                self.Ui, self.Uo, self.Uf, self.Um,
                self.bi, self.bo, self.bf, self.bm,
                self.Vi, self.Vo, self.Vf,
                self.Z, self.bz]

    def get_output(self, X):
        def _fprop(Xi, last_h, last_m):
            gate_i = T.nnet.sigmoid(T.dot(Xi, self.Wi) + T.dot(last_h, self.Ui) + T.dot(last_m, self.Vi) + self.bi)
            mi = T.tanh(T.dot(Xi, self.Wm) + T.dot(last_h, self.Um) + self.bm)

            gate_f = T.nnet.sigmoid(T.dot(Xi, self.Wf) + T.dot(last_h, self.Uf) + T.dot(last_m, self.Vf) + self.bf)
            m = gate_i*mi + gate_f*last_m

            gate_o = T.nnet.sigmoid(T.dot(Xi, self.Wo) + T.dot(last_h, self.Uo) + T.dot(m, self.Vo) + self.bo)
            h = gate_o * T.tanh(m)

            out = T.dot(h, self.Z) + self.bz
            return h, m, out

        results, updates = theano.scan(fn=_fprop,
                                       outputs_info=[T.zeros((X.shape[0], self.hidden_size)),
                                                     T.zeros((X.shape[0], self.hidden_size)),
                                                     None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.
        Hs, Ms, preact_outputs = results

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
