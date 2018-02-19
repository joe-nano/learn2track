import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from smartlearner.interfaces import Loss

from learn2track.models import GRU
from learn2track.models.layers import LayerDense
from learn2track.neurotools import get_neighborhood_directions
from learn2track.utils import l2distance

floatX = theano.config.floatX


class GRU_Regression(GRU):
    """ A standard GRU model with a regression layer stacked on top of it.
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, activation='tanh', use_previous_direction=False, predict_offset=False,
                 use_layer_normalization=False, drop_prob=0., use_zoneout=False, use_skip_connections=False, neighborhood_radius=None,
                 learn_to_stop=False, seed=1234, **_):
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
        activation : str
            Activation function to apply on the "cell candidate"
        use_previous_direction : bool
            Use the previous direction as an additional input
        predict_offset : bool
            Predict the offset from the previous direction instead (need use_previous_direction).
        use_layer_normalization : bool
            Use LayerNormalization to normalize preactivations and stabilize hidden layer evolution
        drop_prob : float
            Dropout/Zoneout probability for recurrent networks. See: https://arxiv.org/pdf/1512.05287.pdf & https://arxiv.org/pdf/1606.01305.pdf
        use_zoneout : bool
            Use zoneout implementation instead of dropout
        use_skip_connections : bool
            Use skip connections from the input to all hidden layers in the network, and from all hidden layers to the output layer
        neighborhood_radius : float
            Add signal in positions around the current streamline coordinate to the input (with given length in voxel space); None = no neighborhood
        learn_to_stop : bool
            Predict whether the streamline being generated should stop or not
        seed : int
            Random seed used for dropout normalization
        """
        self.neighborhood_radius = neighborhood_radius
        self.model_input_size = input_size
        if self.neighborhood_radius:
            self.neighborhood_directions = get_neighborhood_directions(self.neighborhood_radius)
            # Model input size is increased when using neighborhood
            self.model_input_size = input_size * self.neighborhood_directions.shape[0]

        super().__init__(self.model_input_size, hidden_sizes, activation=activation, use_layer_normalization=use_layer_normalization, drop_prob=drop_prob,
                         use_zoneout=use_zoneout, use_skip_connections=use_skip_connections, seed=seed)
        # Restore input size
        self.input_size = input_size

        self.volume_manager = volume_manager
        self.output_size = output_size
        self.use_previous_direction = use_previous_direction
        self.predict_offset = predict_offset
        self.learn_to_stop = learn_to_stop

        if self.predict_offset:
            assert self.use_previous_direction  # Need previous direction to predict offset.

        # Do not use dropout/zoneout in last hidden layer
        layer_regression_activation = "tanh" if self.predict_offset else "identity"
        output_layer_input_size = sum(self.hidden_sizes) if self.use_skip_connections else self.hidden_sizes[-1]
        self.layer_regression = LayerDense(output_layer_input_size, self.output_size, activation=layer_regression_activation, name="GRU_Regression")
        if self.learn_to_stop:
            # Predict whether a streamline should stop or keep growing
            self.layer_stopping = LayerDense(output_layer_input_size, 1, activation='sigmoid', name="GRU_Regression_stopping")

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)
        if self.learn_to_stop:
            self.layer_stopping.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        hyperparameters['use_previous_direction'] = self.use_previous_direction
        hyperparameters['predict_offset'] = self.predict_offset
        hyperparameters['neighborhood_radius'] = self.neighborhood_radius
        hyperparameters['learn_to_stop'] = self.learn_to_stop
        return hyperparameters

    @property
    def parameters(self):
        all_params = super().parameters + self.layer_regression.parameters
        if self.learn_to_stop:
            all_params += self.layer_stopping.parameters

        return all_params

    def _fprop_step(self, Xi, *args):
        # Xi.shape : (batch_size, 4)    *if self.use_previous_direction, Xi.shape : (batch_size,7)
        # coords + dwi ID (+ previous_direction)

        # coords : streamlines 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        batch_size = Xi.shape[0]
        coords = Xi[:, :4]

        # Repeat coords and apply the neighborhood transformations
        if self.neighborhood_radius:
            # coords.shape : (batch_size*len(neighbors_positions), 4)
            coords = T.repeat(coords, self.neighborhood_directions.shape[0], axis=0)
            coords = T.set_subtensor(coords[:, :3], coords[:, :3] + T.tile(self.neighborhood_directions, (batch_size, 1)))

        # Get diffusion data.
        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = self.volume_manager.eval_at_coords(coords)

        # Concatenate back the neighborhood data into a single input vector
        if self.neighborhood_radius:
            data_at_coords = T.reshape(data_at_coords, (batch_size, self.model_input_size))

        if self.use_previous_direction:
            # previous_direction.shape : (batch_size, 3)
            previous_direction = Xi[:, 4:]
            fprop_input = T.concatenate([data_at_coords, previous_direction], axis=1)
        else:
            fprop_input = data_at_coords

        # Hidden state to be passed to the next GRU iteration (next _fprop call)
        # next_hidden_state.shape : n_layers * (batch_size, layer_size)
        next_hidden_state = super()._fprop(fprop_input, *args)

        # Compute the direction to follow for step (t)
        output_layer_input = T.concatenate(next_hidden_state, axis=-1) if self.use_skip_connections else next_hidden_state[-1]
        regression_out = self.layer_regression.fprop(output_layer_input)

        if self.predict_offset:
            regression_out += previous_direction  # Skip-connection from the previous direction.

        outputs = (regression_out,)

        if self.learn_to_stop:
            stopping_out = self.layer_stopping.fprop(output_layer_input)
            outputs = (stopping_out, regression_out)

        return next_hidden_state + outputs

    def get_output(self, X):
        # X.shape : (batch_size, seq_len, n_features=[4|7])
        # For tractography n_features is (x,y,z) + (dwi_id,) + [previous_direction]

        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        outputs_info = outputs_info_h + [None]
        if self.learn_to_stop:
            outputs_info += [None]

        results, updates = theano.scan(fn=self._fprop_step,
                                       # We want to scan over sequence elements, not the examples.
                                       sequences=[T.transpose(X, axes=(1, 0, 2))],
                                       outputs_info=outputs_info,
                                       non_sequences=self.parameters + self.volume_manager.volumes,
                                       strict=True)

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        # regression_out.shape : (batch_size, seq_len, target_size=3)
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        model_output = self.regression_out

        if self.learn_to_stop:
            self.stopping_out = T.transpose(results[-2], axes=(1, 0, 2))
            model_output = (self.stopping_out, self.regression_out)

        return model_output

    def make_sequence_generator(self, subject_id=0, **_):
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

        new_states = self._fprop_step(symb_x_t, *states_h)
        new_states_h = new_states[:len(self.hidden_sizes)]

        # predictions.shape : (batch_size, target_size)
        predictions = [new_states[-1]]
        if self.learn_to_stop:
            predictions = new_states[-2:]

        f = theano.function(inputs=[symb_x_t] + states_h,
                            outputs=list(predictions) + list(new_states_h))

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
            if self.learn_to_stop:
                stopping = results[0]
                next_x_t = results[1]
                new_states = results[2:]
                output = (next_x_t, stopping)
            else:
                next_x_t = results[0]
                new_states = results[1:]
                output = next_x_t

            return output, new_states

        return _gen


class L2DistanceForSequences(Loss):
    """ Computes the L2 error of the output.

    Notes
    -----
    This loss assumes the regression target at every time step is a vector.
    """
    def __init__(self, model, dataset, normalize_output=False, eps=1e-6, sum_over_timestep=False):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output
        self.eps = eps
        self.sum_over_timestep = sum_over_timestep

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= l2distance(regression_outputs, keepdims=True, eps=self.eps)

        self.samples = regression_outputs

        # loss_per_time_step.shape = (batch_size, seq_len)
        self.loss_per_time_step = l2distance(self.samples, self.dataset.symb_targets, eps=self.eps)
        # loss_per_seq.shape = (batch_size,)
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1)

        if not self.sum_over_timestep:
            self.loss_per_seq /= T.sum(mask, axis=1)

        return self.loss_per_seq


class L2DistanceAndStoppingCriteriaForSequences(Loss):
    """ Computes the L2 error and stopping criteria cross-entropy of the output.

    Notes
    -----
    This loss assumes the regression target at every time step is a vector.
    """
    def __init__(self, model, dataset, normalize_output=False, eps=1e-6, sum_over_timestep=False):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output
        self.eps = eps
        self.sum_over_timestep = sum_over_timestep

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, out_dim)
        stopping_criteria_outputs = model_output[0][:, :, 0]
        regression_outputs = model_output[1]

        regression_targets = self.dataset.symb_targets[:, :, :3]
        stopping_criteria_targets = self.dataset.symb_targets[:, :, 3]

        if self.normalize_output:
            regression_outputs /= l2distance(regression_outputs, keepdims=True, eps=self.eps)

        self.samples = regression_outputs

        l2_loss_per_time_step = l2distance(self.samples, regression_targets, eps=self.eps)
        stopping_cross_entropy_per_time_step = T.nnet.binary_crossentropy(stopping_criteria_outputs, stopping_criteria_targets)

        # loss_per_time_step.shape = (batch_size, seq_len)
        self.loss_per_time_step = l2_loss_per_time_step + stopping_cross_entropy_per_time_step
        # loss_per_seq.shape = (batch_size,)
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1)

        if not self.sum_over_timestep:
            self.loss_per_seq /= T.sum(mask, axis=1)

        return self.loss_per_seq
