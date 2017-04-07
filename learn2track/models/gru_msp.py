import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from os.path import join as pjoin
from smartlearner import utils as smartutils
from smartlearner.interfaces import Loss
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learn2track.models import GRU
from learn2track.models.layers import LayerRegression
from learn2track.utils import logsumexp, l2distance

floatX = theano.config.floatX


class GRU_Multistep_Gaussian(GRU):
    """ A multistep GRU model used to predict multivariate gaussian parameters (means and standard deviations)

    For each target dimension, the model outputs (m) distribution parameters estimates for each prediction horizon up to (k)
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, target_dims, k, m, seed, use_previous_direction=False, use_layer_normalization=False,
                 dropout_prob=0., **_):
        """
        Parameters
        ----------
        volume_manager : :class:`VolumeManger` object
            Used to evaluate the diffusion signal at specific coordinates using multiple subjects
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        target_dims : int
            Number of dimensions of the multivariate gaussian to estimate; the model outputs two distribution parameters for each dimension
        k : int
            Number of steps ahead to predict (the model will predict all steps up to k)
        m : int
            Number of Monte-Carlo samples used to estimate the gaussian parameters
        seed : int
            Random seed to initialize the random noise used for sampling and dropout.
        use_previous_direction : bool
            Use the previous direction as an additional input
        use_layer_normalization : bool
            Use LayerNormalization to normalize preactivations and stabilize hidden layer evolution
        dropout_prob : float
            Dropout probability for recurrent networks. See: https://arxiv.org/pdf/1512.05287.pdf
        """
        super().__init__(input_size, hidden_sizes, use_layer_normalization, dropout_prob, seed)
        self.target_dims = target_dims
        self.target_size = 2 * self.target_dims  # Output distribution parameters mu and sigma for each dimension
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.target_size, normed=False)

        self.volume_manager = volume_manager

        self.k = k
        self.m = m
        self.seed = seed

        self.use_previous_direction = use_previous_direction

        self.srng = MRG_RandomStreams(self.seed)

        if self.dropout_prob:
            self.dropout_matrices[self.layer_regression.name] = self.srng.binomial(size=self.layer_regression.W.shape, n=1, p=1 - self.dropout_prob,
                                                                                   dtype=floatX)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['target_dims'] = self.target_dims
        hyperparameters['target_size'] = self.target_size
        hyperparameters['k'] = self.k
        hyperparameters['m'] = self.m
        hyperparameters['seed'] = self.seed
        hyperparameters['use_previous_direction'] = self.use_previous_direction
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop_step(self, Xi, *args):
        # Xi.shape : (batch_size, 4)    *if self.use_previous_direction, Xi.shape : (batch_size,7)
        # coords + dwi ID (+ previous_direction)

        # coords : streamlines 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        coords = Xi[:, :4]

        batch_size = Xi.shape[0]

        if self.k > 1:
            # Random noise used for sampling at each step (t+2)...(t+k)
            # epsilon.shape : (K-1, batch_size, target_dimensions)
            epsilon = self.srng.normal((self.k - 1, batch_size, self.target_dims))

        # Get diffusion data.
        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = self.volume_manager.eval_at_coords(coords)

        if self.use_previous_direction:
            # previous_direction.shape : (batch_size, 3)
            previous_direction = Xi[:, 4:]
            fprop_input = T.concatenate([data_at_coords, previous_direction], axis=1)
        else:
            fprop_input = data_at_coords

        # Hidden state to be passed to the next GRU iteration (next _fprop call)
        # next_hidden_state.shape : n_layers * (batch_size, layer_size)
        next_hidden_state = super()._fprop(fprop_input, *args)

        # Compute the distribution parameters for step (t)
        # distribution_params.shape : (batch_size, target_size)
        distribution_params = self._predict_distribution_params(next_hidden_state[-1])
        # k_distribution_params = T.set_subtensor(k_distribution_params[:, 0, :, :], distribution_params)
        k_distribution_params = [distribution_params]

        sample_hidden_state = next_hidden_state

        for k in range(1, self.k):
            # Sample an input for the next step
            # sample_directions.shape : (batch_size, target_dimensions)
            sample_directions = self.get_stochastic_samples(distribution_params, epsilon[k - 1])

            # Follow *unnormalized* direction and get diffusion data at the new location.
            coords = T.concatenate([coords[:, :3] + sample_directions, coords[:, 3:]], axis=1)

            data_at_coords = self.volume_manager.eval_at_coords(coords)

            if self.use_previous_direction:
                # previous_direction.shape : (batch_size, 3)
                previous_direction = sample_directions
                fprop_input = T.concatenate([data_at_coords, previous_direction], axis=1)
            else:
                fprop_input = data_at_coords

            # Compute the sample distribution parameters for step (t+k)
            sample_hidden_state = super()._fprop(fprop_input, *sample_hidden_state)
            distribution_params = self._predict_distribution_params(sample_hidden_state[-1])
            k_distribution_params += [distribution_params]

        k_distribution_params = T.stack(k_distribution_params, axis=1)

        return next_hidden_state + (k_distribution_params,)

    @staticmethod
    def get_stochastic_samples(distribution_parameters, noise):
        # distribution_parameters.shape : (batch_size, [seq_len], target_size)
        # distribution_params[0] = [mu_x, mu_y, mu_z, std_x, std_y, std_z]

        # noise.shape : (batch_size, target_dims)

        mu = distribution_parameters[..., :3]
        sigma = distribution_parameters[..., 3:6]

        samples = mu + noise * sigma

        return samples

    @staticmethod
    def get_max_component_samples(distribution_parameters):
        # distribution_parameters.shape : (batch_size, [seq_len], target_size)
        # distribution_params[0] = [mu_x, mu_y, mu_z, std_x, std_y, std_z]
        mean = distribution_parameters[..., :3]
        return mean

    def _predict_distribution_params(self, hidden_state):
        # regression layer outputs an array [mean_x, mean_y, mean_z, log(std_x), log(std_y), log(std_z)]
        # regression_output.shape : (batch_size, target_size)
        dropout_W = self.dropout_matrices[self.layer_regression.name] if self.dropout_prob else None
        regression_output = self.layer_regression.fprop(hidden_state, dropout_W)

        # Use T.exp to retrieve a positive sigma
        distribution_params = T.set_subtensor(regression_output[..., 3:6], T.exp(regression_output[..., 3:6]))

        # distribution_params.shape : (batch_size, target_size)
        return distribution_params

    def get_output(self, X):
        # X.shape : (batch_size, seq_len, n_features=4)
        # For tractography n_features is (x,y,z) + (dwi_id,)

        # Repeat Xs to compute M sample sequences for each input
        # inputs.shape : (batch_size*M, seq_len, n_features)
        inputs = T.repeat(X, self.m, axis=0)

        # outputs_info_h.shape : n_layers * (batch_size*M, layer_size)
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((inputs.shape[0], hidden_size)))

        # results.shape : n_layers * (seq_len, batch_size*M, layer_size), (seq_len, batch_size*M, K, target_size)
        results, updates = theano.scan(fn=self._fprop_step, # We want to scan over sequence elements, not the examples.
                                       sequences=[T.transpose(inputs, axes=(1, 0, 2))], outputs_info=outputs_info_h + [None],
                                       non_sequences=self.parameters + self.volume_manager.volumes, strict=True)

        self.graph_updates = updates

        # Put back the examples so they are in the first dimension
        # transposed.shape : (batch_size*M, seq_len, K, target_size)
        transposed = T.transpose(results[-1], axes=(1, 0, 2, 3))

        # Split the M sample sequences into a new dimension
        # reshaped.shape : (batch_size, M, seq_len, K, target_size)
        reshaped = T.reshape(transposed, (X.shape[0], self.m, X.shape[1], self.k, self.target_size))

        # Transpose the output to get the M sequences dimension in the right place
        # regression_out.shape : (batch_size, seq_len, K, M, target_size)
        regression_out = T.transpose(reshaped, (0, 2, 3, 1, 4))

        return regression_out

    def make_sequence_generator(self, subject_id=0, use_max_component=False):
        """ Makes functions that return the prediction for x_{t+1} for every
        sequence in the batch given x_{t} and the current state of the model h^{l}_{t}.

        Parameters
        ----------
        subject_id : int, optional
            ID of the subject from which its diffusion data will be used. Default: 0.
        """

        # Build the sequence generator as a theano function.
        states_h = []
        for i in range(len(self.hidden_sizes)):
            state_h = T.matrix(name="layer{}_state_h".format(i))
            states_h.append(state_h)

        symb_x_t = T.matrix(name="x_t")

        # Temporarily set $k$ to one.
        k_bak = self.k
        self.k = 1

        new_states = self._fprop_step(symb_x_t, *states_h)
        new_states_h = new_states[:len(self.hidden_sizes)]

        # model_output.shape : (batch_size, K=1, target_size)
        model_output = new_states[-1]

        distribution_params = model_output[:, 0, :]

        if use_max_component:
            predictions = self.get_max_component_samples(distribution_params)
        else:
            # Sample value from distribution
            srng = MRG_RandomStreams(seed=1234)

            batch_size = symb_x_t.shape[0]
            noise = srng.normal((batch_size, self.target_dims))

            # predictions.shape : (batch_size, target_dims)
            predictions = self.get_stochastic_samples(distribution_params, noise)

        f = theano.function(inputs=[symb_x_t] + states_h,
                            outputs=[predictions] + list(new_states_h))

        self.k = k_bak  # Restore original $k$.

        def _gen(x_t, states, previous_direction=None):
            """ Returns the prediction for x_{t+1} for every
                sequence in the batch given x_{t} and the current states
                of the model h^{l}_{t}.

            Parameters
            ----------
            x_t : ndarray with shape (batch_size, 3)
                Streamline coordinate (x, y, z).
            states : list of 2D array of shape (batch_size, hidden_size)
                Currrent states of the network.
            previous_direction : ndarray with shape (batch_size, 3)
                If using previous direction, these should be added to the input

            Returns
            -------
            next_x_t : ndarray with shape (batch_size, 3)
                Directions to follow.
            new_states : list of 2D array of shape (batch_size, hidden_size)
                Updated states of the network after seeing x_t.
            """
            # Append the DWI ID of each sequence after the 3D coordinates.
            subject_ids = np.array([subject_id] * len(x_t), dtype=floatX)[:, None]

            if not self.use_previous_direction:
                x_t = np.c_[x_t, subject_ids]
            else:
                x_t = np.c_[x_t, subject_ids, previous_direction]

            results = f(x_t, *states)
            next_x_t = results[0]
            new_states = results[1:]
            return next_x_t, new_states

        return _gen

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
        state = np.load(pjoin(loaddir, 'state.npz'))

        self.srng.rstate[:] = state['_srng_rstate']

        for state_update, saved_state in zip(self.srng.state_updates, state["_srng_state_updates"]):
            state_update[0].set_value(saved_state)


class MultistepMultivariateGaussianNLL(Loss):
    """ Compute multistep loss for a multivariate gaussian distribution using a monte-carlo estimate of the likelihood
    """

    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self._graph_updates = {}
        self.m = np.float32(model.m)
        self.target_dims = np.float32(model.target_dims)

    def _get_updates(self):
        return {}

    def _compute_losses(self, model_output):
        # model_output.shape : (batch_size, seq_len, K, M, target_size)
        # self.dataset.symb_targets.shape = (batch_size, seq_len+K-1, target_dims)

        # mask.shape : (batch_size, seq_len) or None
        mask = self.dataset.symb_mask

        # mu.shape = (batch_size, seq_len, K, M, target_dims)
        mu = model_output[:, :, :, :, 0:3]

        # sigma.shape = (batch_size, seq_len, K, M, target_dims)
        sigma = model_output[:, :, :, :, 3:6]

        # Stack K targets for each input (sliding window style)
        # targets.shape = (batch_size, seq_len, K, target_dims)
        targets = T.stack([self.dataset.symb_targets[:, i:(-self.model.k + i + 1) or None] for i in range(self.model.k)], axis=2)

        # Add new axis for sum over M
        # targets.shape = (batch_size, seq_len, K, 1, target_dims)
        targets = targets[:, :, :, None, :]

        # For monitoring the L2 error of using $mu$ as the predicted direction (should be comparable to MICCAI's work).
        normalized_mu = mu[:, :, 0, 0] / l2distance(mu[:, :, 0, 0], keepdims=True, eps=1e-8)
        normalized_targets = targets[:, :, 0, 0] / l2distance(targets[:, :, 0, 0], keepdims=True, eps=1e-8)
        self.L2_error_per_item = T.sqrt(T.sum(((normalized_mu - normalized_targets) ** 2), axis=2))
        if mask is not None:
            self.mean_sqr_error = T.sum(self.L2_error_per_item * mask, axis=1) / T.sum(mask, axis=1)
        else:
            self.mean_sqr_error = T.mean(self.L2_error_per_item, axis=1)

        # Likelihood of multivariate gaussian (n dimensions) is :
        # ((2 \pi)^D |\Sigma|)^{-1/2} exp(-1/2 (x - \mu)^T \Sigma^-1 (x - \mu))
        # We suppose a diagonal covariance matrix, so we have :
        #   => |\Sigma| = \prod_n \sigma_n^2
        #   => (x - \mu)^T \Sigma^-1 (x - \mu) = \sum_n ((x_n - \mu_n) / \sigma_n)^2
        m_log_likelihoods = -np.float32((self.target_dims/2.) * np.log(2 * np.pi)) + T.sum(-T.log(sigma) - 0.5 * T.sqr((targets - mu) / sigma), axis=4)

        # k_losses_per_timestep.shape : (batch_size, seq_len, K)
        self.k_losses_per_timestep = T.log(self.m) - logsumexp(m_log_likelihoods, axis=3, keepdims=False)

        # loss_per_timestep.shape : (batch_size, seq_len)
        self.loss_per_time_step = T.mean(self.k_losses_per_timestep, axis=2)

        # Average over sequence steps.
        # k_nlls_per_seq.shape :(batch_size, K)
        if mask is not None:
            self.k_losses_per_seq = T.sum(self.k_losses_per_timestep * mask[:, :, None], axis=1) / T.sum(mask, axis=1, keepdims=True)
        else:
            self.k_losses_per_seq = T.mean(self.k_losses_per_timestep, axis=1)

        # Average over K
        # loss_per_seq.shape :(batch_size,)
        self.loss_per_seq = T.mean(self.k_losses_per_seq, axis=1)
        return self.loss_per_seq


class MultistepMultivariateGaussianExpectedValueL2Distance(Loss):
    """ Compute the L2 distance loss based on the expected value
    """

    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self._graph_updates = {}
        self.m = np.float32(model.m)
        self.target_dims = np.float32(model.target_dims)

    def _get_updates(self):
        return {}

    def _compute_losses(self, model_output):
        # model_output.shape : (batch_size, seq_len, K, M, target_size)
        # self.dataset.symb_targets.shape = (batch_size, seq_len+K-1, target_dims)

        # targets.shape = (batch_size, seq_len, 3)
        targets = self.dataset.symb_targets[:, :-self.model.k + 1 or None, :]

        # mask.shape : (batch_size, seq_len)
        mask = self.dataset.symb_mask

        # samples.shape : (batch_size, seq_len, 3)
        # T.squeeze(.) should remove the K=1 and M=1 dimensions
        self.samples = self.model.get_max_component_samples(T.squeeze(model_output))

        # loss_per_time_step.shape = (batch_size, seq_len)
        self.loss_per_time_step = l2distance(self.samples, targets)
        # loss_per_seq.shape = (batch_size,)
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1) / T.sum(mask, axis=1)

        return self.loss_per_seq
