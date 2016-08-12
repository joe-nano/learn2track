from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from smartlearner.interfaces import Loss
from smartlearner.utils import sharedX
import smartlearner.initializers as initer

from learn2track.models.layers import LayerRegression
from learn2track.interpolation import eval_volume_at_3d_coordinates_in_theano
from learn2track.models import GRU

floatX = theano.config.floatX


class GRU_Regression(GRU):
    """ A standard GRU model with a regression layer stacked on top of it.
    """
    def __init__(self, dwis, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        dwis : list of 4D arrays with shape (width, height, depth, nb_gradients)
            Diffusion signal as weighted images.
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        output_size : int
            Number of units the regression layer should have.
        """
        super().__init__(input_size, hidden_sizes)

        self.dwis = [sharedX(dwi, name='dwi', keep_on_cpu=False) for dwi in dwis]

        # Precompute strides that will be used in the interpolation.
        self.dwi_strides = []
        for dwi in self.dwis:
            shapes = T.cast(dwi.shape[:-1], dtype=floatX)
            strides = T.concatenate([T.ones((1,)), T.cumprod(shapes[::-1])[:-1]], axis=0)[::-1]
            self.dwi_strides.append(strides.eval())

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
        # coords : streamlines 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        coords = Xi

        # Get diffusion data.
        data_at_coords = []
        for i, (dwi, dwi_stride) in enumerate(zip(self.dwis, self.dwi_strides)):
            selection = T.eq(coords[:, 3], i).nonzero()[0]  # Theano's way of doing: coords[:, 3] == i
            selected_coords = coords[selection, :3]
            data_at_selected_coords = eval_volume_at_3d_coordinates_in_theano(dwi, selected_coords, strides=dwi_stride)
            data_at_coords.append(data_at_selected_coords)

        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = T.concatenate(data_at_coords, axis=0)

        # Hidden state to be passed to the next GRU iteration (next _fprop call)
        # next_hidden_state.shape : n_layers * (batch_size, layer_size)
        next_hidden_state = super()._fprop(data_at_coords, *args)

        # Compute the direction to follow for step (t)
        regression_out = self.layer_regression.fprop(next_hidden_state[-1])

        return next_hidden_state + (regression_out,)

    def get_output(self, X):
        # X.shape : (batch_size, seq_len, n_features=4)
        # For tractography n_features is (x,y,z) + (dwi_id,)

        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       # We want to scan over sequence elements, not the examples.
                                       sequences=[T.transpose(X, axes=(1, 0, 2))],
                                       outputs_info=outputs_info_h + [None],
                                       non_sequences=self.parameters + self.dwis,
                                       strict=True)

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        # regression_out.shape : (batch_size, seq_len, target_size=3)
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        return self.regression_out

    def use(self, X):
        directions = self.get_output(X)
        return directions


class GRU_Regression_OLD(GRU):
    """ A standard GRU model with a regression layer stacked on top of it.
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


class L2DistanceForSequences(Loss):
    """ Computes the L2 error of the output.

    Notes
    -----
    This loss assumes the regression target at every time step is a vector.
    """
    def __init__(self, model, dataset, normalize_output=False):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= (T.sqrt(T.sum(regression_outputs**2, axis=2, keepdims=True) + 1e-8))

        # L2_errors_per_time_step.shape = (batch_size,)
        self.L2_errors_per_time_step = T.sqrt(T.sum(((regression_outputs - self.dataset.symb_targets)**2), axis=2))
        # avg_L2_error_per_seq.shape = (batch_size,)
        self.avg_L2_error_per_seq = T.sum(self.L2_errors_per_time_step*mask, axis=1) / T.sum(mask, axis=1)

        return self.avg_L2_error_per_seq
