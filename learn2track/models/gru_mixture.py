import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from smartlearner.interfaces import Loss
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learn2track.models.gru_regression import GRU_Regression
from learn2track.models.layers import LayerRegression
from learn2track.utils import logsumexp, softmax

floatX = theano.config.floatX


class GRU_Mixture(GRU_Regression):
    """ A GRU_Regression model with the output size computed for a mixture of gaussians, using a diagonal covariance matrix
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, n_gaussians, **_):
        """
        Parameters
        ----------
        volume_manager : :class:`VolumeManger` object
            Use to evaluate the diffusion signal at specific coordinates.
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        output_size : int
            Number of units the regression layer should have.
        n_gaussians : int
            Number of gaussians in the mixture
        """
        super(GRU_Regression, self).__init__(input_size, hidden_sizes)
        self.volume_manager = volume_manager
        self.n_gaussians = n_gaussians

        assert output_size == 3  # Only 3-dimensional target is supported for now
        self.output_size = output_size

        self.layer_regression_size = sum([n_gaussians,  # Mixture weights
                                          n_gaussians * output_size,  # Means
                                          n_gaussians * output_size])  # Stds
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.layer_regression_size)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['n_gaussians'] = self.n_gaussians
        hyperparameters['layer_regression_size'] = self.layer_regression_size
        return hyperparameters

    def get_mixture_parameters(self, regression_output):
        batch_size = regression_output.shape[0]
        mixture_weights = T.nnet.softmax(regression_output[:, :self.n_gaussians])
        means = T.reshape(regression_output[:, self.n_gaussians:self.n_gaussians * 4], (batch_size, self.n_gaussians, 3))
        stds = T.reshape(T.exp(regression_output[:, self.n_gaussians * 4:self.n_gaussians * 7]), (batch_size, self.n_gaussians, 3))

        return mixture_weights, means, stds

    def _get_mixture_samples(self, srng, mixture_weights, means, stds):
        batch_size = mixture_weights.shape[0]
        xs = T.arange(0, batch_size)

        choices = T.argmax(srng.multinomial(n=1, pvals=mixture_weights), axis=1)

        # means[0] : [[mean_x1, mean_y1, mean_z1], ..., [mean_xn, mean_yn, mean_zn]]

        # mu.shape : (batch_size, 3)
        mu = means[xs, choices]

        # sigma.shape : (batch_size, 3)
        sigma = stds[xs, choices]

        noise = srng.normal((batch_size, 3))
        samples = mu + sigma * noise

        return samples

    def seq_next(self, x_t, subject_ids=None):
        """ Returns the prediction for x_{t+1} for every sequence in the batch.

        Parameters
        ----------
        x_t : ndarray with shape (batch_size, 3)
            Streamline coordinate (x, y, z).
        subject_ids : ndarray with shape (batch_size, 1), optional
            ID of the subject from which its diffusion data will be used. Default: [0]*len(x_t)
        """
        if subject_ids is None:
            subject_ids = np.array([0] * len(x_t), dtype=floatX)[:, None]

        # Append the DWI ID of each sequence after the 3D coordinates.
        x_t = np.c_[x_t, subject_ids]

        if self._gen is None:
            # Build theano function and cache it.
            self.seq_reset(batch_size=len(x_t))

            symb_x_t = T.TensorVariable(type=T.TensorType("floatX", [False] * x_t.ndim), name='x_t')
            symb_x_t.tag.test_value = x_t

            states = self.states_h + [0]
            new_states = self._fprop_step(symb_x_t, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]

            # regression_output.shape : (batch_size, target_size)
            regression_output = new_states[-1]
            mixture_params = self.get_mixture_parameters(regression_output)

            srng = MRG_RandomStreams(1234)
            predictions = self._get_mixture_samples(srng, *mixture_params)

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]

            self._gen = theano.function([symb_x_t], predictions, updates=updates)

        return self._gen(x_t)


class MultivariateGaussianMixtureNLL(Loss):
    """ Computes the likelihood of a multivariate gaussian mixture
    """

    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self.n = model.n_gaussians
        self.d = model.output_size

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, regression_layer_size)
        regression_outputs = model_output

        mixture_weights, means, stds = self._get_mixture_parameters(regression_outputs)

        # means.shape : (batch_size, seq_len, n_gaussians, 3)

        # mean_*.shape : (batch_size, seq_len, n_gaussians)
        mean_x = means[:, :, :, 0]
        mean_y = means[:, :, :, 1]
        mean_z = means[:, :, :, 2]

        # std_*.shape : (batch_size, seq_len, n_gaussians)
        std_x = stds[:, :, :, 0]
        std_y = stds[:, :, :, 1]
        std_z = stds[:, :, :, 2]

        # target_*.shape : (batch_size, seq_len, 1)
        target_x = self.dataset.symb_targets[:, :, 0, None]
        target_y = self.dataset.symb_targets[:, :, 1, None]
        target_z = self.dataset.symb_targets[:, :, 2, None]

        tg_x_c = (target_x - mean_x) / std_x
        tg_y_c = (target_y - mean_y) / std_y
        tg_z_c = (target_z - mean_z) / std_z

        log_prefix = T.log(mixture_weights) - np.float32((self.d / 2.) * np.log(2 * np.pi)) - T.log(std_x) - T.log(std_y) - T.log(std_z)
        square_mahalanobis_dist = -0.5 * (tg_x_c ** 2 + tg_y_c ** 2 + tg_z_c ** 2)

        # nll_per_timestep.shape : (batch_size, seq_len)
        self.nll_per_timestep = - logsumexp(log_prefix + square_mahalanobis_dist, axis=2)

        # nll_per_seq.shape : (batch_size,)
        self.nll_per_seq = T.sum(self.nll_per_timestep * mask, axis=1) / T.sum(mask, axis=1)

        return self.nll_per_seq

    def _get_mixture_parameters(self, regression_output):
        batch_size, seq_len = regression_output.shape[:2]

        gaussian_3d_params_shape = (batch_size, seq_len, self.n, 3)

        mixture_weights = softmax(regression_output[:, :, :self.n], axis=2)
        means = T.reshape(regression_output[:, :, self.n:self.n * 4], gaussian_3d_params_shape)
        stds = T.reshape(T.exp(regression_output[:, :, self.n * 4:self.n * 7]), gaussian_3d_params_shape)

        return mixture_weights, means, stds

class MultivariateGaussianMixtureExpectedValueL2Distance(Loss):
    """ Computes the likelihood of a multivariate gaussian mixture
    """

    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self.n = model.n_gaussians
        self.d = model.output_size

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, regression_layer_size)
        regression_outputs = model_output

        mixture_weights, means, stds = self._get_mixture_parameters(regression_outputs)

        # mixture_weights.shape : (batch_size, seq_len, n_gaussians)
        # means.shape : (batch_size, seq_len, n_gaussians, 3)

        # expected_value.shape : (batch_size, seq_len, 3)
        expected_value = T.sum(mixture_weights[:, :, :, None] * means, axis=2)

        # L2_errors_per_time_step.shape = (batch_size, seq_len)
        self.L2_errors_per_time_step = T.sqrt(T.sum(((expected_value - self.dataset.symb_targets)**2), axis=2))
        # avg_L2_error_per_seq.shape = (batch_size,)
        self.avg_L2_error_per_seq = T.sum(self.L2_errors_per_time_step*mask, axis=1) / T.sum(mask, axis=1)

        return self.avg_L2_error_per_seq

    def _get_mixture_parameters(self, regression_output):
        batch_size, seq_len = regression_output.shape[:2]

        gaussian_3d_params_shape = (batch_size, seq_len, self.n, 3)

        mixture_weights = softmax(regression_output[:, :, :self.n], axis=2)
        means = T.reshape(regression_output[:, :, self.n:self.n * 4], gaussian_3d_params_shape)
        stds = T.reshape(T.exp(regression_output[:, :, self.n * 4:self.n * 7]), gaussian_3d_params_shape)

        return mixture_weights, means, stds