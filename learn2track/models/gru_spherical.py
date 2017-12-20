import numpy as np
import theano
import theano.tensor as T
from smartlearner.interfaces import Loss
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learn2track.models.gru_regression import GRU_Regression
from learn2track.models.layers import LayerRegression, LayerDense
from learn2track.neurotools import get_neighborhood_directions
from learn2track.utils import logsumexp, softmax, l2distance

floatX = theano.config.floatX


class GRU_SphericalHarmonics(GRU_Regression):
    """ A GRU_Regression model with the output size computed for spherical harmonics

        We need real-valued anti-symmetric spherical harmonics, therefore we use only odd order spherical harmonics, as defined by (Descoteaux, M..
        "High Angular Resolution Diffusion MRI: from Local Estimation to Segmentation and Tractography.", 2008.)

        l = 0,1,3,5,...; m = -l, ..., 0, ..., +l
        j(l,m) = {1 if l = m = 0; (l^2 + l + 4)/2 + m otherwise}
        R = (1/2)(L+1)(L+2) + 1
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, sh_order=6, use_previous_direction=False, use_layer_normalization=False, drop_prob=0.,
                 use_zoneout=False, use_skip_connections=False, neighborhood_radius=False, learn_to_stop=False, seed=1234, **_):
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
        sh_order : int
            Order of spherical harmonics coefficients used to estimate next streamline direction
        use_previous_direction : bool
            Use the previous direction as an additional input
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

        super(GRU_Regression, self).__init__(self.model_input_size, hidden_sizes, use_layer_normalization=use_layer_normalization, drop_prob=drop_prob,
                                             use_zoneout=use_zoneout, use_skip_connections=use_skip_connections, seed=seed)
        # Restore input size
        self.input_size = input_size

        self.volume_manager = volume_manager

        assert output_size == 3  # Only 3-dimensional target is supported for now
        self.output_size = output_size

        self.sh_order = sh_order

        self.use_previous_direction = use_previous_direction

        self.learn_to_stop = learn_to_stop

        self.layer_regression_size = int((self.sh_order + 1) * (self.sh_order + 2) // 2 + 1)
        output_layer_input_size = sum(self.hidden_sizes) if self.use_skip_connections else self.hidden_sizes[-1]

        # Do not use dropout/zoneout in last hidden layer
        self.layer_regression = LayerRegression(output_layer_input_size, self.layer_regression_size)
        if self.learn_to_stop:
            # Predict whether a streamline should stop or keep growing
            self.layer_stopping = LayerDense(output_layer_input_size, 1, activation='sigmoid', name="GRU_SphericalHarmonics_stopping")

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['sh_order'] = self.sh_order
        return hyperparameters

    @staticmethod
    def _get_stochastic_samples(srng, sh_coeffs):
        # SH to SF
        # TODO: Conversion from SH to SF

        # Normalize SF
        SF = None

        # Multinomial from SF
        multinomial = srng.multinomial(n=1, pvals=SF)
        index = T.argmax(multinomial, axis=-1)

        # Get vector on the sphere corresponding to index
        # TODO: Fetch sphere vector
        samples = None

        return samples

    @staticmethod
    def _get_max_component_samples(sh_coeffs):
        # SH to SF
        # TODO: Conversion from SH to SF
        SF = None

        index = T.argmax(SF, axis=-1)

        # Get vector on the sphere corresponding to index
        # TODO: Fetch sphere vector
        samples = None

        return samples

    def make_sequence_generator(self, subject_id=0, use_max_component=False):
        """ Makes functions that returns the prediction for x_{t+1} for every
        sequence in the batch given x_{t} and the current state of the model h^{l}_{t}.

        Parameters
        ----------
        subject_id : int, optional
            ID of the subject from which its diffusion data will be used. Default: 0.
        use_max_component : bool, optional
            Use the maximum of the probability distribution instead of sampling values
        """

        # Build the sequence generator as a theano function.
        states_h = []
        for i in range(len(self.hidden_sizes)):
            state_h = T.matrix(name="layer{}_state_h".format(i))
            states_h.append(state_h)

        symb_x_t = T.matrix(name="x_t")

        new_states = self._fprop_step(symb_x_t, *states_h)
        new_states_h = new_states[:len(self.hidden_sizes)]

        # sh_coefficients.shape : (batch_size, target_size)
        sh_coefficients = new_states[-1]

        if use_max_component:
            samples = self._get_max_component_samples(sh_coefficients)
        else:
            srng = MRG_RandomStreams(1234)
            samples = self._get_stochastic_samples(srng, sh_coefficients)

        if self.learn_to_stop:
            stopping = new_states[-2]
            predictions = [stopping, samples]
        else:
            predictions = [samples]

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
            next_states : list of 2D array of shape (batch_size, hidden_size)
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


class SHReconstruction(Loss):
    """ Computes the reconstruction error using SH coefficients
    """

    def __init__(self, model, dataset, sum_over_timestep=False):
        super().__init__(model, dataset)
        self.sum_over_timestep = sum_over_timestep

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # sh_coeffs.shape = (batch_size, seq_length, regression_layer_size)
        sh_coeffs = model_output

        # targets.shape : (batch_size, seq_len, 3)
        targets = self.dataset.symb_targets

        # B should be pre-computed
        # TODO: Pre-compute B matrix
        B = None
        reconstruction = T.dot(B, sh_coeffs)

        # loss_per_timestep.shape : (batch_size, seq_len)
        self.loss_per_time_step = l2distance(targets, reconstruction)

        # loss_per_seq.shape : (batch_size,)
        # loss_per_seq is the log probability for each sequence
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1)

        if not self.sum_over_timestep:
            # loss_per_seq is the average log probability for each sequence
            self.loss_per_seq /= T.sum(mask, axis=1)

        return self.loss_per_seq


class SHReconstructionAndStoppingCriteria(Loss):
    """ Computes the reconstruction error using SH coefficients + stopping criteria cross-entropy
    """

    def __init__(self, model, dataset, sum_over_timestep=False, gamma=1.0):
        super().__init__(model, dataset)
        self.sum_over_timestep = sum_over_timestep
        self.gamma = gamma

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # sh_coeffs.shape = (batch_size, seq_length, regression_layer_size)
        stopping_criteria_outputs = model_output[0][:, :, 0]
        sh_coeffs = model_output[1]

        # targets.shape : (batch_size, seq_len, 3)
        targets = self.dataset.symb_targets
        stopping_criteria_targets = self.dataset.symb_targets[:, :, 3]

        # B should be pre-computed
        # TODO: Pre-compute B matrix
        B = None
        reconstruction = T.dot(B, sh_coeffs)

        recontruction_error_per_time_step = l2distance(targets, reconstruction)
        stopping_cross_entropy_per_time_step = T.nnet.binary_crossentropy(stopping_criteria_outputs, stopping_criteria_targets)

        # loss_per_timestep.shape : (batch_size, seq_len)
        # self.gamma should be used to balance the two loss terms. Consider tweaking this hyperparameter if training goes wrong.
        self.loss_per_time_step = recontruction_error_per_time_step + self.gamma * stopping_cross_entropy_per_time_step

        # loss_per_seq.shape : (batch_size,)
        # loss_per_seq is the log probability for each sequence
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1)

        if not self.sum_over_timestep:
            # loss_per_seq is the average log probability for each sequence
            self.loss_per_seq /= T.sum(mask, axis=1)

        return self.loss_per_seq


