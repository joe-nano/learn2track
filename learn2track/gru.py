from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from learn2track.layers import LayerGRU, LayerRegression, LayerSoftmax

from smartlearner.interfaces import Model
from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer


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
            self.states_h[i].set_value(np.zeros_like(self.states_h[i].get_value()))

    def seq_next(self, input):
        """ Returns the next element in every sequence of the batch. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + self.states_m
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


class GRU_Softmax(GRU):
    """ A standard GRU model with a softmax layer stacked on top of it.

    The output of this model is a distribution.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
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
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + [None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates = updates
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


class GRU_Regression(GRU):
    """ A standard GRU model regression layer stacked on top of it.

    The output of this model is normalized vector.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        output_size : int
            Number of units the regression layer should have.
        """
        # super().__init__(input_size, hidden_sizes)
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
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + [None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        return self.regression_out

    def use(self, X):
        directions = self.get_output(X)
        return directions


class GRU_Hybrid(GRU):
    """ A standard GRU model regression layer stacked on top of it, then a Softmax layer.

    The output of this model is probabilities distribution over the targets.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
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
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + [None, None],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates = updates
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
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]

            self._gen = theano.function([X], output, updates=updates)

        return self._gen(input)

    def use(self, X):
        """ Returns the directions from the sphere. """
        idx = T.argmax(self.get_output(X), axis=2)
        return self.directions[idx]
