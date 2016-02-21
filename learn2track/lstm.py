import os
from os.path import join as pjoin

import pickle
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from learn2track import factories

from smartlearner.interfaces import Model
from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer


class LayerDense(object):
    def __init__(self, input_size, output_size, activation="identity", name="Dense"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Regression output weights and biases
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        out = self.activation_fct(preactivation)
        return out


class LayerLSTM(object):
    def __init__(self, input_size, hidden_size, activation="tanh", name="LSTM"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Input weights (i:input, o:output, f:forget, m:memory)
        self.Wi = sharedX(value=np.zeros((input_size, hidden_size)), name=self.name+'_Wi')
        self.Wo = sharedX(value=np.zeros((input_size, hidden_size)), name=self.name+'_Wo')
        self.Wf = sharedX(value=np.zeros((input_size, hidden_size)), name=self.name+'_Wf')
        self.Wm = sharedX(value=np.zeros((input_size, hidden_size)), name=self.name+'_Wm')

        # Biases (i:input, o:output, f:forget, m:memory)
        self.bi = sharedX(value=np.zeros(hidden_size), name=self.name+'_bi')
        self.bo = sharedX(value=np.zeros(hidden_size), name=self.name+'_bo')
        self.bf = sharedX(value=np.zeros(hidden_size), name=self.name+'_bf')
        self.bm = sharedX(value=np.zeros(hidden_size), name=self.name+'_bm')

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        self.Ui = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Ui')
        self.Uo = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uo')
        self.Uf = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uf')
        self.Um = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Um')

        # Memory weights (i:input, o:output, f:forget, m:memory)
        self.Vi = sharedX(value=np.ones(hidden_size), name=self.name+'_Vi')
        self.Vo = sharedX(value=np.ones(hidden_size), name=self.name+'_Vo')
        self.Vf = sharedX(value=np.ones(hidden_size), name=self.name+'_Vf')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for param in [self.Wi, self.Wo, self.Wf, self.Wm]:
            weights_initializer(param)

        for param in [self.Ui, self.Uo, self.Uf, self.Um]:
            weights_initializer(param)

    @property
    def parameters(self):
        return [self.Wi, self.Wo, self.Wf, self.Wm,
                self.Ui, self.Uo, self.Uf, self.Um,
                self.bi, self.bo, self.bf, self.bm,
                self.Vi, self.Vo, self.Vf]

    def fprop(self, Xi, last_h, last_m):
        # TODO: replace sigmoid by ReLU?
        gate_i = T.nnet.sigmoid(T.dot(Xi, self.Wi) + T.dot(last_h, self.Ui) + last_m*self.Vi + self.bi)
        mi = T.tanh(T.dot(Xi, self.Wm) + T.dot(last_h, self.Um) + self.bm)

        gate_f = T.nnet.sigmoid(T.dot(Xi, self.Wf) + T.dot(last_h, self.Uf) + last_m*self.Vf + self.bf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(T.dot(Xi, self.Wo) + T.dot(last_h, self.Uo) + m*self.Vo + self.bo)
        h = gate_o * self.activation_fct(m)

        return h, m


class LayerLSTMFast(object):
    def __init__(self, input_size, hidden_size, activation="tanh", name="LSTM"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        self.W = sharedX(value=np.zeros((input_size, 4*hidden_size)), name=self.name+'_W')

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = sharedX(value=np.zeros(4*hidden_size), name=self.name+'_b')

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        self.U = sharedX(value=np.zeros((hidden_size, 4*hidden_size)), name=self.name+'_U')

        # Memory weights (i:input, o:output, f:forget, m:memory)
        self.Vi = sharedX(value=np.ones(hidden_size), name=self.name+'_Vi')
        self.Vo = sharedX(value=np.ones(hidden_size), name=self.name+'_Vo')
        self.Vf = sharedX(value=np.ones(hidden_size), name=self.name+'_Vf')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)
        weights_initializer(self.U)

    @property
    def parameters(self):
        return [self.W, self.U, self.b,
                self.Vi, self.Vo, self.Vf]

    def fprop(self, Xi, last_h, last_m):
        def slice_(x, no):
            if type(no) is str:
                no = ['i', 'o', 'f', 'm'].index(no)
            return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

        Xi = (T.dot(Xi, self.W) + self.b)  # SPEEDUP: compute this outside for all timestep at once.
        preactivation = Xi + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(slice_(preactivation, 'i') + last_m*self.Vi)
        mi = self.activation_fct(slice_(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(slice_(preactivation, 'f') + last_m*self.Vf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(slice_(preactivation, 'o') + m*self.Vo)
        h = gate_o * self.activation_fct(m)

        return h, m


class LayerLSTMFaster(object):
    def __init__(self, input_size, hidden_size, activation="tanh", name="LSTM"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        self.W = sharedX(value=np.zeros((input_size, 4*hidden_size)), name=self.name+'_W')

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = sharedX(value=np.zeros(4*hidden_size), name=self.name+'_b')

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        self.U = sharedX(value=np.zeros((hidden_size, 4*hidden_size)), name=self.name+'_U')

        # Memory weights (i:input, o:output, f:forget, m:memory)
        self.Vi = sharedX(value=np.ones(hidden_size), name=self.name+'_Vi')
        self.Vo = sharedX(value=np.ones(hidden_size), name=self.name+'_Vo')
        self.Vf = sharedX(value=np.ones(hidden_size), name=self.name+'_Vf')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)
        weights_initializer(self.U)

    @property
    def parameters(self):
        return [self.W, self.U, self.b,
                self.Vi, self.Vo, self.Vf]

    def fprop(self, Xi, last_h, last_m):
        def slice_(x, no):
            if type(no) is str:
                no = ['i', 'o', 'f', 'm'].index(no)
            return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

        preactivation = Xi + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(slice_(preactivation, 'i') + last_m*self.Vi)
        mi = self.activation_fct(slice_(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(slice_(preactivation, 'f') + last_m*self.Vf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(slice_(preactivation, 'o') + m*self.Vo)
        h = gate_o * self.activation_fct(m)

        return h, m


class LayerRegression(object):
    def __init__(self, input_size, output_size, normed=True, name="Regression"):

        self.input_size = input_size
        self.output_size = output_size
        self.normed = normed
        self.name = name

        # Regression output weights and biases
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, X):
        out = T.dot(X, self.W) + self.b
        # Normalize the output vector.
        if self.normed:
            out /= (T.sqrt(T.sum(out**2, axis=1, keepdims=True) + 1e-8))

        return out


class LayerSoftmax(object):
    def __init__(self, input_size, output_size, name="Softmax"):

        self.input_size = input_size
        self.output_size = output_size
        self.name = name

        # Regression output weights and biases
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        # The softmax function, applied to a matrix, computes the softmax values row-wise.
        out = T.nnet.softmax(preactivation)
        return out


class Stackable(Model):
    def __init__(self):
        self.layers = []
        self._prev = None
        self._next = None

    def __add__(self, layer):
        self._next = layer
        layer._prev = self


class LSTM(Model):
    """ A standard LSTM model with no output layer.

    See LSTM_softmax or LSTM_regression for implementations with an output layer.

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

        self.layers_lstm = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers_lstm.append(LayerLSTM(last_hidden_size, hidden_size, name="LSTM{}".format(i)))
            last_hidden_size = hidden_size

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for layer_lstm in self.layers_lstm:
            layer_lstm.initialize(weights_initializer)

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
        for layer_lstm in self.layers_lstm:
            parameters += layer_lstm.parameters

        return parameters

    def _fprop(self, Xi, *args):
        layers_h = []
        layers_m = []

        input = Xi
        for i, layer_lstm in enumerate(self.layers_lstm):
            last_h, last_m = args[i], args[len(self.layers_lstm)+i]
            h, m = layer_lstm.fprop(input, last_h, last_m)
            layers_h.append(h)
            layers_m.append(m)
            input = h

        return tuple(layers_h) + tuple(layers_m)

    def get_output(self, X):
        outputs_info_h = []
        outputs_info_m = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))
            outputs_info_m.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + outputs_info_m,
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        self.h = T.transpose(results[0], axes=(1, 0, 2))
        return self.h

    def seq_squeeze(self, tokeep):
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(self.states_h[i].get_value()[tokeep])
            self.states_m[i].set_value(self.states_m[i].get_value()[tokeep])

    def seq_reset(self, batch_size=None):
        """ Start a new batch of sequences. """
        if self._gen is None:
            self.states_h = []
            self.states_m = []
            for i, hidden_size in enumerate(self.hidden_sizes):
                self.states_h.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_h".format(i)))
                self.states_m.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_m".format(i)))

        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(np.zeros_like(self.states_h[i].get_value()))
            self.states_m[i].set_value(np.zeros_like(self.states_m[i].get_value()))

    def seq_next(self, input):
        """ Returns the next element in every sequence of the batch. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + self.states_m
            new_states = self._fprop(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]
            new_states_m = new_states[len(self.hidden_sizes):-1]
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]
                updates[self.states_m[i]] = new_states_m[i]

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

class LSTM(Model):
    """ A standard LSTM model with no output layer.

    See LSTM_softmax or LSTM_regression for implementations with an output layer.

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

        self.layers_lstm = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers_lstm.append(LayerLSTMFast(last_hidden_size, hidden_size, name="LSTM{}".format(i)))
            last_hidden_size = hidden_size

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for layer_lstm in self.layers_lstm:
            layer_lstm.initialize(weights_initializer)

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
        for layer_lstm in self.layers_lstm:
            parameters += layer_lstm.parameters

        return parameters

    def _fprop(self, Xi, *args):
        layers_h = []
        layers_m = []

        input = Xi
        for i, layer_lstm in enumerate(self.layers_lstm):
            last_h, last_m = args[i], args[len(self.layers_lstm)+i]
            h, m = layer_lstm.fprop(input, last_h, last_m)
            layers_h.append(h)
            layers_m.append(m)
            input = h

        return tuple(layers_h) + tuple(layers_m)

    def get_output(self, X):
        outputs_info_h = []
        outputs_info_m = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))
            outputs_info_m.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + outputs_info_m,
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        self.h = T.transpose(results[0], axes=(1, 0, 2))
        return self.h

    def seq_squeeze(self, tokeep):
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(self.states_h[i].get_value()[tokeep])
            self.states_m[i].set_value(self.states_m[i].get_value()[tokeep])

    def seq_reset(self, batch_size=None):
        """ Start a new batch of sequences. """
        if self._gen is None:
            self.states_h = []
            self.states_m = []
            for i, hidden_size in enumerate(self.hidden_sizes):
                self.states_h.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_h".format(i)))
                self.states_m.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_m".format(i)))

        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(np.zeros_like(self.states_h[i].get_value()))
            self.states_m[i].set_value(np.zeros_like(self.states_m[i].get_value()))

    def seq_next(self, input):
        """ Returns the next element in every sequence of the batch. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + self.states_m
            new_states = self._fprop(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]
            new_states_m = new_states[len(self.hidden_sizes):-1]
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]
                updates[self.states_m[i]] = new_states_m[i]

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

class LSTMFaster(Model):
    """ A standard LSTM model with no output layer.

    See LSTM_softmax or LSTM_regression for implementations with an output layer.

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
        self.mask = None

        self.input_size = input_size
        self.hidden_sizes = [hidden_sizes] if type(hidden_sizes) is int else hidden_sizes

        self.layers_lstm = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            if i == 0:
                layer = LayerLSTMFaster(last_hidden_size, hidden_size, name="LSTM{}".format(i))
            else:
                layer = LayerLSTMFast(last_hidden_size, hidden_size, name="LSTM{}".format(i))

            self.layers_lstm.append(layer)
            last_hidden_size = hidden_size

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for layer_lstm in self.layers_lstm:
            layer_lstm.initialize(weights_initializer)

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
        for layer_lstm in self.layers_lstm:
            parameters += layer_lstm.parameters

        return parameters

    def _fprop(self, Xi, *args):
        input = Xi
        layers_h, layers_m = [], []
        for i, layer_lstm in enumerate(self.layers_lstm):
            last_h, last_m = args[i], args[len(self.layers_lstm)+i]
            h, m = layer_lstm.fprop(input, last_h, last_m)
            layers_h.append(h)
            layers_m.append(m)
            input = h

        return tuple(layers_h) + tuple(layers_m)

    def _fprop_masked(self, Xi, mask, *args):
        outs = list(self._fprop(Xi, *args))

        layers_h_masked, layers_m_masked = [], []
        for i, layer_lstm in enumerate(self.layers_lstm):
            last_h, last_m = args[i], args[len(self.layers_lstm)+i]
            h, m = outs[i], outs[len(self.layers_lstm)+i]
            h = mask[:, None] * h + (1 - mask[:, None]) * last_h
            m = mask[:, None] * m + (1 - mask[:, None]) * last_m
            layers_h_masked.append(h)
            layers_m_masked.append(m)

        return tuple(layers_h_masked) + tuple(layers_m_masked)

    def get_output(self, X):
        outputs_info_h = []
        outputs_info_m = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))
            outputs_info_m.append(T.zeros((X.shape[0], hidden_size)))

        X = T.dot(X, self.layers_lstm[0].W) + self.layers_lstm[0].b

        if self.mask is not None:
            results, updates = theano.scan(fn=self._fprop_masked,
                                           outputs_info=outputs_info_h + outputs_info_m,
                                           # We want to scan over sequence elements not the examples, so we transpose.
                                           sequences=[T.transpose(X, axes=(1, 0, 2)),
                                                      self.mask.T])

        else:
            results, updates = theano.scan(fn=self._fprop,
                                           outputs_info=outputs_info_h + outputs_info_m,
                                           # We want to scan over sequence elements not the examples, so we transpose.
                                           sequences=[T.transpose(X, axes=(1, 0, 2))])

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        self.h = T.transpose(results[0], axes=(1, 0, 2))
        return self.h

    def seq_squeeze(self, tokeep):
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(self.states_h[i].get_value()[tokeep])
            self.states_m[i].set_value(self.states_m[i].get_value()[tokeep])

    def seq_reset(self, batch_size=None):
        """ Start a new batch of sequences. """
        if self._gen is None:
            self.states_h = []
            self.states_m = []
            for i, hidden_size in enumerate(self.hidden_sizes):
                self.states_h.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_h".format(i)))
                self.states_m.append(sharedX(np.zeros((batch_size, hidden_size)), name="layer{}_state_m".format(i)))

        for i, hidden_size in enumerate(self.hidden_sizes):
            self.states_h[i].set_value(np.zeros_like(self.states_h[i].get_value()))
            self.states_m[i].set_value(np.zeros_like(self.states_m[i].get_value()))

    def seq_next(self, input):
        """ Returns the next element in every sequence of the batch. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + self.states_m
            new_states = self._fprop(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]
            new_states_m = new_states[len(self.hidden_sizes):-1]
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]
                updates[self.states_m[i]] = new_states_m[i]

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


class LSTM_Softmax(LSTM):
    """ A standard LSTM model with a softmax layer stacked on top of it.

    The output of this model is a distribution.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each LSTM should have.
        output_size : int
            Number of units the softmax layer should have.
        """
        super().__init__(input_size, hidden_sizes)
        self.output_size = output_size
        self.layer_softmax = LayerSoftmax(self.hidden_sizes[-1], self.output_size)

        from dipy.data import get_sphere
        # self.directions = sharedX(get_sphere("repulsion" + str(self.output_size)).vertices.astype(theano.config.floatX))
        self.directions = get_sphere("repulsion" + str(self.output_size)).vertices.astype(theano.config.floatX)
        self.shared_directions = sharedX(self.directions)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_softmax.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_softmax.parameters

    def _fprop(self, Xi, *args):
        outputs = super()._fprop(Xi, *args)
        last_layer_h = outputs[len(self.hidden_sizes)-1]
        probs = self.layer_softmax.fprop(last_layer_h)
        return outputs + (probs,)

    def get_output(self, X):
        outputs_info_h = []
        outputs_info_m = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))
            outputs_info_m.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + outputs_info_m + [None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        self.probs = T.transpose(results[-1], axes=(1, 0, 2))
        return self.probs

    def seq_next(self, input):
        next_timestep = super().seq_next(input)
        idx = np.argmax(next_timestep, axis=1)
        return self.directions[idx]

    def use(self, X):
        idx = T.argmax(self.get_output(X), axis=2)
        return self.directions[idx]


class LSTM_SoftmaxWithFeaturesExtraction(LSTM_Softmax):
    """ A LSTM model with a fully connected layer for extracting
    features as the first layer and a softmax layer as the output layer.

    The output of this model is a distribution.
    """
    def __init__(self, input_size, features_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        features_size : int
            Size of the features space (i.e. the first layer).
        hidden_sizes : list of int
            Numbers of hidden units each LSTM layer.
        output_size : int
            Number of units the softmax layer.
        """
        self.features_size = features_size
        super().__init__(self.features_size, hidden_sizes, output_size)
        self.input_size = input_size  # Override
        self.layer_dense = LayerDense(self.input_size, self.features_size)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_dense.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['features_size'] = self.features_size
        return hyperparameters

    @property
    def parameters(self):
        return self.layer_dense.parameters + super().parameters

    def _fprop(self, Xi, *args):
        h0 = self.layer_dense.fprop(Xi)
        outputs = super()._fprop(h0, *args)
        return outputs


class LSTM_Regression(LSTM):
    """ A standard LSTM model regression layer stacked on top of it.

    The output of this model is normalized vector.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each LSTM should have.
        output_size : int
            Number of units the regression layer should have.
        """
        super().__init__(input_size, hidden_sizes)
        self.output_size = output_size
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.output_size)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop(self, Xi, *args):
        outputs = super()._fprop(Xi, *args)
        last_layer_h = outputs[len(self.hidden_sizes)-1]
        regression_out = self.layer_regression.fprop(last_layer_h)
        return outputs + (regression_out,)

    def get_output(self, X):
        outputs_info_h = []
        outputs_info_m = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))
            outputs_info_m.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + outputs_info_m + [None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        return self.regression_out

    def use(self, X):
        directions = self.get_output(X)
        return directions


class LSTM_RegressionWithFeaturesExtraction(LSTM_Regression):
    """ A LSTM model with a fully connected layer for extracting
    features as the first layer and a regression layer as the output layer.

    The output of this model is normalized vectors.
    """
    def __init__(self, input_size, features_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        features_size : int
            Size of the features space (i.e. the first layer).
        hidden_sizes : list of int
            Numbers of hidden units each LSTM layer.
        output_size : int
            Number of units the regression layer.
        """
        self.features_size = features_size
        super().__init__(self.features_size, hidden_sizes, output_size)
        self.input_size = input_size  # Override
        self.layer_dense = LayerDense(self.input_size, self.features_size)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_dense.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['features_size'] = self.features_size
        return hyperparameters

    @property
    def parameters(self):
        return self.layer_dense.parameters + super().parameters

    def _fprop(self, Xi, *args):
        h0 = self.layer_dense.fprop(Xi)
        outputs = super()._fprop(h0, *args)
        return outputs


class LSTM_Hybrid(LSTM):
    """ A standard LSTM model regression layer stacked on top of it, then a Softmax layer.

    The output of this model is probabilities distribution over the targets.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each LSTM should have.
        output_size : int
            Number of units the softmax layer should have.
        """
        super().__init__(input_size, hidden_sizes)
        self.output_size = output_size
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], 3, normed=True)

        from dipy.data import get_sphere
        self.sphere = get_sphere("repulsion" + str(self.output_size))
        self.sphere_directions = self.sphere.vertices.astype(theano.config.floatX)
        # self.shared_directions = sharedX(self.directions)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop_regression(self, Xi, *args):
        outputs = super()._fprop(Xi, *args)
        last_layer_h = outputs[len(self.hidden_sizes)-1]
        regression_out = self.layer_regression.fprop(last_layer_h)
        return outputs + (regression_out,)

    def _fprop(self, Xi, *args):
        outputs = self._fprop_regression(Xi, *args)
        regression_out = outputs[-1]

        # Map regression output to one of the allowed directions, then take the softmax.
        preactivation = T.dot(regression_out, self.sphere_directions.T)
        probs = T.nnet.softmax(preactivation)  # The softmax function, applied to a matrix, computes the softmax values row-wise.
        return outputs + (probs,)

    def get_output(self, X):
        outputs_info_h = []
        outputs_info_m = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))
            outputs_info_m.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + outputs_info_m + [None, None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates.update(updates)
        # Put back the examples so they are in the first dimension.
        self.directions = T.transpose(results[-2], axes=(1, 0, 2))
        self.probs = T.transpose(results[-1], axes=(1, 0, 2))
        return self.probs

    def seq_next(self, input):
        """ Returns the next *unnormalized* directions. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + self.states_m
            new_states = self._fprop_regression(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]
            new_states_m = new_states[len(self.hidden_sizes):-1]
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]
                updates[self.states_m[i]] = new_states_m[i]

            self._gen = theano.function([X], output, updates=updates)

        return self._gen(input)

    def use(self, X):
        """ Returns the directions from the sphere. """
        idx = T.argmax(self.get_output(X), axis=2)
        return self.directions[idx]
