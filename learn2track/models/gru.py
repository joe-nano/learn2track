import os
from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

from smartlearner.interfaces import Model
from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer

from learn2track.layers import LayerGRU, LayerRegression, LayerSoftmax, LayerDense
from learn2track.tasks import DecayingVariable
from learn2track.interpolation import eval_volume_at_3d_coordinates_in_theano


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
    """ A standard GRU model with a regression layer stacked on top of it.

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


class GRU_RegressionWithBinaryClassification(GRU):
    """ A standard GRU model with a regression layer stacked on top of it
    combined with a fully connected layer with a sigmoid non-linearity
    to learn when to stop.

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
        super().__init__(input_size, hidden_sizes)
        self.output_size = output_size
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.output_size)
        self.stopping_layer = LayerDense(self.hidden_sizes[-1] + input_size, 1, activation="sigmoid", name="stopping")

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)
        self.stopping_layer.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters + self.stopping_layer.parameters

    def _fprop(self, Xi, Xi_plus1, *args):
        outputs = super()._fprop(Xi, *args)
        last_layer_h = outputs[len(self.hidden_sizes)-1]
        regression_out = self.layer_regression.fprop(last_layer_h)
        stopping = self.stopping_layer.fprop(T.concatenate([last_layer_h, Xi_plus1], axis=1))

        return outputs + (regression_out, stopping)

    def get_output(self, X):
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + [None, None],
                                       sequences=[{"input": T.transpose(X, axes=(1, 0, 2)),  # We want to scan over sequence elements, not the examples.
                                                   "taps": [0, 1]}],
                                       n_steps=X.shape[1]-1)

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        self.regression_out = T.transpose(results[-2], axes=(1, 0, 2))
        self.stopping = T.transpose(results[-1], axes=(1, 0, 2))

        return self.regression_out, self.stopping

    def use(self, X):
        directions = self.get_output(X)
        return directions


class GRU_RegressionWithScheduledSampling(GRU):
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
        super().__init__(input_size, hidden_sizes)
        self.output_size = output_size
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.output_size)
        self.scheduled_sampling_rate = DecayingVariable(name="scheduled_sampling_rate")

    @property
    def tasks(self):
        return [self.scheduled_sampling_rate]

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
        # We assumed that the last 3 values of Xi are the target direction at the previous
        # timestep (i.e. --append-previous-direction as been used)
        last_direction = args[-1]  # Predicted direction at last timestep
        rate = self.scheduled_sampling_rate.var
        Xi = T.set_subtensor(Xi[:, -3:],
                             rate*Xi[:, -3:] + (1-rate)*last_direction)

        outputs = super()._fprop(Xi, *args)
        last_layer_h = outputs[len(self.hidden_sizes)-1]
        regression_out = self.layer_regression.fprop(last_layer_h)
        return outputs + (regression_out,)

    def get_output(self, X):
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        last_direction = T.zeros((X.shape[0], 3))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h + [last_direction],
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        return self.regression_out

    def seq_next(self, input):
        """ Returns the next element in every sequence of the batch. """
        if self._gen is None:
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False]*input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + [0]
            new_states = self._fprop(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]
            output = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]

            self._gen = theano.function([X], output, updates=updates)

        return self._gen(input)

    def use(self, X):
        directions = self.get_output(X)
        return directions

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

            states = self.states_h
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


class GRU_Multistep_Gaussian(GRU):
    """ A multistep GRU model used to predict multivariate gaussian parameters (means and standard deviations)

    For each target dimension, the model outputs (m) distribution parameters estimates for each prediction horizon up to (k)
    """

    def __init__(self, dwi, input_size, hidden_sizes, target_size, k, m, seed, **_):
        """
        Parameters
        ----------
        dwi : 4D array with shape (width, height, depth, nb. diffusion directions)
            Diffusion signal as weighted images.
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        target_size : int
            Dimension of the multivariate gaussian to estimate; the model outputs two distribution parameters for each dimension of the target
        k : int
            Number of steps ahead to predict (the model will predict all steps up to k)
        m : int
            Number of Monte-Carlo samples used to estimate the gaussian parameters
        seed : int
            Random seed to initialize the random noise used for sampling
        """
        super().__init__(input_size, hidden_sizes)
        self.target_size = target_size  # Output distribution parameters mu and sigma for each dimension
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], 2 * self.target_size, normed=False)

        self.dwi = sharedX(dwi, name='dwi', keep_on_cpu=False)

        # Precompute strides that will be used in the interpolation.
        shapes = T.cast(self.dwi.shape[:-1], dtype=theano.config.floatX)
        strides = T.concatenate([T.ones((1,)), T.cumprod(shapes[::-1])[:-1]], axis=0)[::-1]
        self.dwi_strides = strides.eval()

        self.k = k
        self.m = m
        self.seed = seed

        self.srng = MRG_RandomStreams(self.seed)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['target_size'] = self.target_size
        hyperparameters['k'] = self.k
        hyperparameters['m'] = self.m
        hyperparameters['seed'] = self.seed
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop(self, Xi, *args):
        # Xi : coordinates in a 3D volume.
        # Xi.shape : (batch_size, 3)
        # args.shape : n_layers * (batch_size, layer_size)
        batch_size = Xi.shape[0]
        coords = Xi

        if self.k > 1:
            # Random noise used for sampling at each step (t+2)...(t+k)
            # epsilon.shape : (K, batch_size, target_size)
            epsilon = self.srng.normal((self.k - 1, batch_size, self.target_size))

        # Object to hold the distribution parameters at each prediction horizon
        # k_distribution_params.shape : (batch_size, K, target_size, 2)
        # k_distribution_params = T.zeros((batch_size, self.k, self.target_size, 2))

        # Get diffusion data.
        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = eval_volume_at_3d_coordinates_in_theano(self.dwi, coords, strides=self.dwi_strides)

        # Hidden state to be passed to the next GRU iteration (next _fprop call)
        # next_hidden_state.shape : n_layers * (batch_size, layer_size)
        next_hidden_state = super()._fprop(data_at_coords, *args)

        # Compute the distribution parameters for step (t)
        # distribution_params.shape : (batch_size, target_size, 2)
        distribution_params = self._predict_distribution_params(next_hidden_state[-1])
        # k_distribution_params = T.set_subtensor(k_distribution_params[:, 0, :, :], distribution_params)
        k_distribution_params = [distribution_params]

        sample_hidden_state = next_hidden_state

        for k in range(1, self.k):
            # Sample an input for the next step
            # sample_input.shape : (batch_size, target_size)
            sample_directions = self._get_sample(distribution_params, epsilon[k - 1])

            # Follow *unnormalized* direction and get diffusion data at the new location.
            coords = coords + sample_directions
            data_at_coords = eval_volume_at_3d_coordinates_in_theano(self.dwi, coords, strides=self.dwi_strides)

            # Compute the sample distribution parameters for step (t+k)
            sample_hidden_state = super()._fprop(data_at_coords, *sample_hidden_state)
            distribution_params = self._predict_distribution_params(sample_hidden_state[-1])
            # k_distribution_params = T.set_subtensor(k_distribution_params[:, k, :, :], distribution_params)
            k_distribution_params += [distribution_params]

        k_distribution_params = T.stack(k_distribution_params, axis=1)

        return next_hidden_state + (k_distribution_params,)

    @staticmethod
    def _get_sample(distribution_parameters, noise):
        # distribution_parameters.shape : (batch_size, target_size, 2)
        # noise.shape : (batch_size, target_size)
        mu = distribution_parameters[:, :, 0]
        # Use T.exp to retrieve a positive sigma
        sigma = T.exp(distribution_parameters[:, :, 1])
        return mu + noise * sigma

    def _predict_distribution_params(self, hidden_state):
        # regression layer outputs an array [mu_1, log(sigma_1), mu_2, log(sigma_2), mu_3, log(sigma_3)] for each batch example
        # regression_output.shape : (batch_size, target_size, 2)
        regression_output = T.reshape(self.layer_regression.fprop(hidden_state), (hidden_state.shape[0], self.target_size, 2))

        # Use T.exp to retrieve a positive sigma
        # distribution_params = T.set_subtensor(regression_output[:, :, 1], T.exp(regression_output[:, :, 1]))
        distribution_params = regression_output
        # distribution_params.shape : (batch_size, target_size, 2)
        return distribution_params

    def get_output(self, X):
        # X.shape : (batch_size, seq_len, n_features)

        # Repeat Xs to compute M sample sequences for each input
        # inputs.shape : (batch_size*M, seq_len, n_features)
        inputs = T.repeat(X, self.m, axis=0)

        # outputs_info_h.shape : n_layers * (batch_size*M, layer_size)
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((inputs.shape[0], hidden_size)))

        # results.shape : n_layers * (seq_len, batch_size*M, layer_size), (seq_len, batch_size*M, K, target_size, 2)
        results, updates = theano.scan(fn=self._fprop,
                                       sequences=[T.transpose(inputs, axes=(1, 0, 2))],  # We want to scan over sequence elements, not the examples.
                                       outputs_info=outputs_info_h + [None],
                                       non_sequences=self.parameters + [self.dwi],
                                       strict=True)

        self.graph_updates = updates

        # Put back the examples so they are in the first dimension
        # transposed.shape : (batch_size*M, seq_len, K, target_size, 2)
        transposed = T.transpose(results[-1], axes=(1, 0, 2, 3, 4))

        # Split the M sample sequences into a new dimension
        # reshaped.shape : (batch_size, M, seq_len, K, target_size, 2)
        reshaped = T.reshape(transposed, (X.shape[0], self.m, X.shape[1], self.k, self.target_size, 2))

        # Transpose the output to get the M sequences dimension in the right place
        # regression_out.shape : (batch_size, seq_len, K, M, target_size, 2)
        regression_out = T.transpose(reshaped, (0, 2, 3, 1, 4, 5))

        return regression_out

    # def use(self, X):
    #     # output.shape : (batch_size, seq_len, K, M, target_size, 2)
    #     output = self.get_output(X)

    #     # Sample inputs for mean estimation
    #     epsilon = self.srng.normal((X.shape[0], X.shape[1], self.k, self.m, self.target_size))
    #     means = output[:, :, :, :, :, 0]
    #     stds = output[:, :, :, :, :, 1]

    #     # samples.shape : (batch_size, seq_len, K, M, target_size)
    #     samples = means + epsilon * stds

    #     # predictions.shape : (batch_size, seq_len, K, target_size)
    #     predictions = T.mean(samples, axis=3)

    #     return predictions

    def seq_next(self, input):
        """ Returns the next (t+1) prediction in every sequence of the batch.

            Note: self.k will be fixed to 1 in order to avoid useless computations from (t+2) to (t+k).
        """
        if self._gen is None:
            k_bak = self.k
            self.k = 1  # Temporarily set $k$ to one.
            self.seq_reset(batch_size=len(input))

            X = T.TensorVariable(type=T.TensorType("floatX", [False] * input.ndim), name='X')
            X.tag.test_value = input

            states = self.states_h + [0]
            new_states = self._fprop(X, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]

            # output.shape : (batch_size, K, target_size, 2)
            output = new_states[-1]

            # next_step_predictions.shape : (batch_size, target_size)
            mu = output[:, 0, :, 0]
            next_step_predictions = mu

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]

            self._gen = theano.function([X], next_step_predictions, updates=updates)
            self.k = k_bak  # Restore original $k$.

        return self._gen(input)

    @classmethod
    def create(cls, path, **kwargs):
        loaddir = pjoin(path, cls.__name__)
        hyperparams = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        hyperparams.update(kwargs)
        model = cls(**hyperparams)
        model.load(path)
        return model

    def save(self, path):
        super().save(path)

        savedir = smartutils.create_folder(pjoin(path, type(self).__name__))
        state = {
            "version": 1,
            "_srng_rstate": self.srng.rstate,
            "_srng_state_updates": [state_update[0].get_value() for state_update in self.srng.state_updates]}

        np.savez(pjoin(savedir, "state.npz"), **state)

    def load(self, path):
        super().load(path)

        loaddir = pjoin(path, type(self).__name__)
        # TODO: remove the following file check (only there for backward compatibility).
        if os.path.isfile(pjoin(loaddir, 'state.npz')):
            state = np.load(pjoin(loaddir, 'state.npz'))

            self.srng.rstate[:] = state['_srng_rstate']

            for state_update, saved_state in zip(self.srng.state_updates, state["_srng_state_updates"]):
                state_update[0].set_value(saved_state)
