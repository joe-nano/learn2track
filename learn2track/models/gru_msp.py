import os
from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams

from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer
from smartlearner.interfaces import Loss

from learn2track.models.layers import LayerRegression
from learn2track.interpolation import eval_volume_at_3d_coordinates_in_theano
from learn2track.utils import logsumexp

from learn2track.models import GRU


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


class MultistepMultivariateGaussianLossForSequences(Loss):
    """ Compute multistep loss for a multivariate gaussian distribution using a monte-carlo estimate of the likelihood
    """
    def __init__(self, model, dataset, nb_samples, target_size):
        super().__init__(model, dataset)
        self._graph_updates = {}
        self.nb_samples = np.float32(nb_samples)
        self.target_size = np.float32(target_size)

    def getstate(self):
        state = {"version": 1, "__name__": type(self).__name__}

        return state

    def setstate(self, state):
        pass

    def _get_updates(self):
        return {}

    def _compute_losses(self, model_output):
        # model_output.shape : shape : (batch_size, seq_len, K, M, target_size, 2)
        # self.dataset.symb_targets.shape = (batch_size, seq_len, K, target_size)

        # mask.shape : (batch_size, seq_len) or None
        mask = self.dataset.symb_mask

        # mu.shape = (batch_size, seq_len, K, M, target_size)
        mu = model_output[:, :, :, :, :, 0]

        # sigma.shape = (batch_size, seq_len, K, M, target_size)
        # Use T.exp to retrieve a positive sigma
        sigma = T.exp(model_output[:, :, :, :, :, 1])

        # targets.shape = (batch_size, seq_len, K, 1, target_size)
        targets = self.dataset.symb_targets[:, :, :, None, :]

        # For monitoring the L2 error of using $mu$ as the predicted direction (should be comparable to MICCAI's work).
        normalized_mu = mu[:, :, 0, 0] / T.sqrt(T.sum(mu[:, :, 0, 0]**2, axis=2, keepdims=True) + 1e-8)
        normalized_targets = targets[:, :, 0, 0] / T.sqrt(T.sum(targets[:, :, 0, 0]**2, axis=2, keepdims=True) + 1e-8)
        self.L2_error_per_item = T.sqrt(T.sum(((normalized_mu - normalized_targets)**2), axis=2))
        if mask is not None:
            self.mean_sqr_error = T.sum(self.L2_error_per_item*mask, axis=1) / T.sum(mask, axis=1)
        else:
            self.mean_sqr_error = T.mean(self.L2_error_per_item, axis=1)

        # Likelihood of multivariate gaussian (n dimensions) is :
        # ((2 \pi)^n |\Sigma|)^{-1/2} exp(-1/2 (x - \mu)^T \Sigma^-1 (x - \mu))
        # We suppose a diagonal covariance matrix, so we have :
        #   => |\Sigma| = \prod_n \sigma_n^2
        #   => (x - \mu)^T \Sigma^-1 (x - \mu) = \sum_n ((x_n - \mu_n) / \sigma_n)^2
        likelihood = -0.5 * (self.target_size * np.float32(np.log(2 * np.pi)) + T.sum(2 * T.log(sigma) + T.sqr((targets - mu) / sigma), axis=4))

        # k_nlls_per_timestep.shape :(batch_size, seq_len, K)
        self.k_nlls_per_timestep = T.log(self.nb_samples) - logsumexp(likelihood, axis=3, keepdims=False)

        # Average over sequence steps.
        # k_nlls_per_seq.shape :(batch_size, K)
        if mask is not None:
            self.k_nlls_per_seq = T.sum(self.k_nlls_per_timestep * mask[:, :, None], axis=1) / T.sum(mask, axis=1, keepdims=True)
        else:
            self.k_nlls_per_seq = T.mean(self.k_nlls_per_timestep, axis=1)

        # Average over K
        # nlls.shape :(batch_size,)
        self.nlls = T.mean(self.k_nlls_per_seq, axis=1)
        return self.nlls
