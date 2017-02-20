import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from learn2track.models import FFNN
from learn2track.models.layers import LayerDense
from smartlearner.interfaces.loss import Loss

from learn2track.utils import l2distance

floatX = theano.config.floatX


class FFNN_Regression(FFNN):
    """ A standard FFNN model with a regression layer stacked on top of it.
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, activation, use_previous_direction=False, predict_offset=False, **_):
        """
        Parameters
        ----------
        volume_manager : :class:`VolumeManger` object
            Use to evaluate the diffusion signal at specific coordinates.
        input_size : int
            Number of units each element X has.
        hidden_sizes : int, list of int
            Number of hidden units each FFNN layer should have.
        output_size : int
            Number of units the regression layer should have.
        activation : str
            Name of the activation function to use in the hidden layers
        use_previous_direction : bool
            Use the previous direction as an additional input
        predict_offset : bool
            Predict the offset from the previous direction instead (need use_previous_direction).
        """
        super().__init__(input_size, hidden_sizes, activation)
        self.volume_manager = volume_manager
        self.output_size = output_size
        self.use_previous_direction = use_previous_direction
        self.predict_offset = predict_offset

        if self.predict_offset:
            assert self.use_previous_direction  # Need previous direction to predict offset.

        layer_regression_activation = "tanh" if self.predict_offset else "identity"
        self.layer_regression = LayerDense(self.hidden_sizes[-1], self.output_size, activation=layer_regression_activation)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        hyperparameters['use_previous_direction'] = self.use_previous_direction
        hyperparameters['predict_offset'] = self.predict_offset
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop(self, Xi, *args):
        # Xi.shape : (batch_size, 4)    *if self.use_previous_direction, Xi.shape : (batch_size,7)
        # coords + dwi ID (+ previous_direction)

        # coords : streamlines 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        coords = Xi[:, :4]

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
        layer_outputs = super()._fprop(fprop_input)

        # Compute the direction to follow for step (t)
        regression_out = self.layer_regression.fprop(layer_outputs[-1])
        if self.predict_offset:
            regression_out += previous_direction  # Skip-connection from the previous direction.

        return layer_outputs + (regression_out,)

    def make_sequence_generator(self, subject_id=0, **_):
        """ Makes functions that return the prediction for x_{t+1} for every
        sequence in the batch given x_{t}.

        Parameters
        ----------
        subject_id : int, optional
            ID of the subject from which its diffusion data will be used. Default: 0.
        """

        # Build the sequence generator as a theano function.
        symb_x_t = T.matrix(name="x_t")

        layer_outputs = self._fprop(symb_x_t)

        # predictions.shape : (batch_size, target_size)
        predictions = layer_outputs[-1]

        f = theano.function(inputs=[symb_x_t], outputs=[predictions])

        def _gen(x_t, states, previous_direction=None):
            """ Returns the prediction for x_{t+1} for every
                sequence in the batch given x_{t}.

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

            results = f(x_t)
            next_x_t = results[-1]

            next_x_t_both_directions = np.stack([next_x_t, -next_x_t], axis=-1)

            next_x_t = next_x_t_both_directions[
                (np.arange(next_x_t_both_directions.shape[0])[:, None]),
                (np.arange(next_x_t_both_directions.shape[1])[None, :]),
                np.argmin(np.linalg.norm(next_x_t_both_directions - previous_direction[:, :, None], axis=1), axis=1)[:, None]]

            # FFNN_Regression is not a recurrent network, return original states
            new_states = states

            return next_x_t, new_states

        return _gen


class CosineSquaredLoss(Loss):
    """ Computes the sine squared error of the angle between the target and the output.

    Notes
    -----
    This loss assumes the regression target is a vector.
    """
    def __init__(self, model, dataset, normalize_output=False, eps=1e-6):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output
        self.eps = eps

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        # regression_outputs.shape = (batch_size, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= l2distance(regression_outputs, keepdims=True, eps=self.eps)

        self.samples = regression_outputs

        # Maximize squared cosine similarity = minimize -cos**2
        # loss_per_time_step.shape = (batch_size,)
        self.loss_per_time_step = -T.square(T.sum(self.samples*self.dataset.symb_targets, axis=1))

        return self.loss_per_time_step


class UndirectedL2Distance(Loss):
    """ Computes the undirected L2 error of the output. (min of the l2distance between output and y/-y)

    Notes
    -----
    This loss assumes the regression target is a vector.
    This should not be used for model training!
    """
    def __init__(self, model, dataset, normalize_output=False, eps=1e-6):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output
        self.eps = eps

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        # regression_outputs.shape = (batch_size, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= l2distance(regression_outputs, keepdims=True, eps=self.eps)

        self.samples = regression_outputs

        # loss_per_time_step.shape = (batch_size,)
        self.loss_per_time_step = T.min(
            T.stack([l2distance(self.samples, self.dataset.symb_targets), l2distance(self.samples, -self.dataset.symb_targets)], axis=1), axis=1)

        return self.loss_per_time_step
