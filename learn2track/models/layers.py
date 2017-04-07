import numpy as np
import theano.tensor as T

from learn2track import factories

from smartlearner.utils import sharedX
import smartlearner.initializers as initer

from learn2track.utils import l2distance


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

    def fprop(self, X, dropout_W=None):
        # dropout_W is a row vector of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]
        preactivation = T.dot(X, W) + self.b
        out = self.activation_fct(preactivation)
        return out


class LayerDenseNormalized(object):
    """
    LayerNormalization applied to dense FFNN layer. See: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, input_size, output_size, activation="identity", name="DenseNormalized", eps=1e-5):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)
        self.eps = eps

        # Regression output weights, biases and gains
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')
        self.g = sharedX(value=np.ones(output_size), name=self.name+'_g')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b, self.g]

    def fprop(self, X, dropout_W=None):
        # dropout_W is a row vector of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]
        units_inputs = T.dot(X, W)

        mean = T.mean(units_inputs, axis=1, keepdims=True)
        std = T.std(units_inputs, axis=1, keepdims=True)

        units_inputs_normalized = (units_inputs - mean) / (std + self.eps)

        out = self.activation_fct(self.g * units_inputs_normalized + self.b)

        return out


class LayerRegression(object):
    def __init__(self, input_size, output_size, normed=False, name="Regression"):

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

    def fprop(self, X, dropout_W=None):
        # dropout_W is a row vector of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]
        out = T.dot(X, W) + self.b
        # Normalize the output vector.
        if self.normed:
            out /= l2distance(out, keepdims=True, eps=1e-8)

        return out


class LayerRegressionNormalized(object):
    """
    LayerNormalization applied to dense FFNN layer. See: https://arxiv.org/abs/1607.06450
    """
    def __init__(self, input_size, output_size, normed=False, name="Regression", eps=1e-5):

        self.input_size = input_size
        self.output_size = output_size
        self.normed = normed
        self.name = name
        self.eps = eps

        # Regression output weights, biases and gains
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')
        self.g = sharedX(value=np.ones(output_size), name=self.name+'_g')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b, self.g]

    def fprop(self, X, dropout_W=None):
        # dropout_W is a row vector of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]
        units_inputs = T.dot(X, W)

        mean = T.mean(units_inputs, axis=1, keepdims=True)
        std = T.std(units_inputs, axis=1, keepdims=True)

        units_inputs_normalized = (units_inputs - mean) / (std + self.eps)

        out = self.g * units_inputs_normalized + self.b

        # Normalize the output vector.
        if self.normed:
            out /= l2distance(out, keepdims=True, eps=1e-8)

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

    def fprop(self, X, dropout_W=None):
        # dropout_W is a row vector of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]
        preactivation = T.dot(X, W) + self.b
        # The softmax function, applied to a matrix, computes the softmax values row-wise.
        out = T.nnet.softmax(preactivation)
        return out


class LayerLstmWithPeepholes(object):
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

        # Peepholes (i:input, o:output, f:forget, m:memory)
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

    def fprop(self, Xi, last_h, last_m, dropout_W=None, dropout_U=None):
        # dropout_W, dropout_U are row vectors of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]

        U = self.U
        if dropout_U:
            U *= dropout_U[:, None]

        def slice_(x, no):
            if type(no) is str:
                no = ['i', 'o', 'f', 'm'].index(no)
            return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

        # SPEEDUP: compute the first linear transformation outside the scan i.e. for all timestep at once.
        # EDIT: I try and didn't see much speedup!
        Xi = (T.dot(Xi, W) + self.b)
        preactivation = Xi + T.dot(last_h, U)

        gate_i = T.nnet.sigmoid(slice_(preactivation, 'i') + last_m*self.Vi)
        mi = self.activation_fct(slice_(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(slice_(preactivation, 'f') + last_m*self.Vf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(slice_(preactivation, 'o') + m*self.Vo)
        h = gate_o * self.activation_fct(m)

        return h, m


class LayerGRU(object):
    """ Gated Recurrent Unit

    References
    ----------
    .. [Chung14] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio
                 "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling",
                 http://arxiv.org/pdf/1412.3555v1.pdf, 2014
    """
    def __init__(self, input_size, hidden_size, activation="tanh", name="GRU"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Input weights (z:update, r:reset)
        # Concatenation of the weights in that order: Wz, Wr, Wh
        self.W = sharedX(value=np.zeros((input_size, 3*hidden_size)), name=self.name+'_W')
        # self.Wh = sharedX(value=np.zeros((input_size, 2*hidden_size)), name=self.name+'_Wh')

        # Biases (z:update, r:reset)
        # Concatenation of the biases in that order: bz, br, bh
        self.b = sharedX(value=np.zeros(3*hidden_size), name=self.name+'_b')
        # self.bh = sharedX(value=np.zeros(hidden_size), name=self.name+'_bh')

        # Recurrence weights (z:update, r:reset)
        # Concatenation of the recurrence weights in that order: Uz, Ur
        self.U = sharedX(value=np.zeros((hidden_size, 2*hidden_size)), name=self.name+'_U')
        self.Uh = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uh')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)
        # weights_initializer(self.Wh)
        weights_initializer(self.U)
        weights_initializer(self.Uh)

    @property
    def parameters(self):
        return [self.W, self.b, self.U, self.Uh]

    def fprop(self, Xi, last_h, dropout_W=None, dropout_U=None, dropout_Uh=None):
        # dropout_W, dropout_U, dropout_Uh are row vectors of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]

        U = self.U
        if dropout_U:
            U *= dropout_U[:, None]

        Uh = self.Uh
        if dropout_Uh:
            U *= dropout_Uh[:, None]

        def slice_(x, no):
            if type(no) is str:
                if no == 'zr':
                    return x[:, :2*self.hidden_size]

                no = ['z', 'r', 'h'].index(no)

            return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

        Xi = (T.dot(Xi, W) + self.b)
        preactivation = slice_(Xi, 'zr') + T.dot(last_h, U)

        gate_z = T.nnet.sigmoid(slice_(preactivation, 'z'))  # Update gate
        gate_r = T.nnet.sigmoid(slice_(preactivation, 'r'))  # Reset gate

        # Candidate activation
        c = self.activation_fct(slice_(Xi, 'h') + T.dot(last_h*gate_r, Uh))
        h = (1-gate_z)*last_h + gate_z*c

        return h


class LayerGruNormalized(object):
    """ Gated Recurrent Unit

    LayerNormalization applied to dense GRU layer. See: https://arxiv.org/abs/1607.06450

    References
    ----------
    .. [Chung14] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio
                 "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling",
                 http://arxiv.org/pdf/1412.3555v1.pdf, 2014
    """
    def __init__(self, input_size, hidden_size, activation="tanh", name="GRU", eps=1e-5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)
        self.eps = eps

        # Input weights (z:update, r:reset)
        # Concatenation of the weights in that order: Wz, Wr, Wh
        self.W = sharedX(value=np.zeros((input_size, 3*hidden_size)), name=self.name+'_W')

        self.b_x = sharedX(value=np.zeros(3 * hidden_size), name=self.name + '_b_x')
        self.b_u = sharedX(value=np.zeros(2 * hidden_size), name=self.name + '_b_u')
        self.b_uh = sharedX(value=np.zeros(hidden_size), name=self.name+'_b_uh')

        self.g_x = sharedX(value=np.ones(3 * hidden_size), name=self.name + '_g_x')
        self.g_u = sharedX(value=np.ones(2*hidden_size), name=self.name+'_g_u')
        self.g_uh = sharedX(value=np.ones(hidden_size), name=self.name+'_g_uh')

        # Recurrence weights (z:update, r:reset)
        # Concatenation of the recurrence weights in that order: Uz, Ur
        self.U = sharedX(value=np.zeros((hidden_size, 2*hidden_size)), name=self.name+'_U')
        self.Uh = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uh')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)
        weights_initializer(self.U)
        weights_initializer(self.Uh)

    @property
    def parameters(self):
        return [self.W, self.b_x, self.b_u, self.b_uh, self.U, self.Uh, self.g_x, self.g_u, self.g_uh]

    def fprop(self, Xi, last_h, dropout_W=None, dropout_U=None, dropout_Uh=None):
        # dropout_W, dropout_U, dropout_Uh are row vectors of inputs to be dropped
        W = self.W
        if dropout_W:
            W *= dropout_W[:, None]

        U = self.U
        if dropout_U:
            U *= dropout_U[:, None]

        Uh = self.Uh
        if dropout_Uh:
            U *= dropout_Uh[:, None]

        def slice_(x, no):
            if type(no) is str:
                if no == 'zr':
                    return x[:, :2*self.hidden_size]

                no = ['z', 'r', 'h'].index(no)

            return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

        def layer_normalize(x, g, b):
            mean = T.mean(x, axis=1, keepdims=True)
            std = T.sqrt(T.var(x, axis=1, keepdims=True) + self.eps)
            x_normalized = (x - mean) / (std + self.eps)
            return g * x_normalized + b

        Xi = layer_normalize(T.dot(Xi, W), self.g_x, self.b_x)
        X_zr = slice_(Xi, 'zr')
        preactivation = X_zr + layer_normalize(T.dot(last_h, U), self.g_u, self.b_u)

        gate_z = T.nnet.sigmoid(slice_(preactivation, 'z'))  # Update gate
        gate_r = T.nnet.sigmoid(slice_(preactivation, 'r'))  # Reset gate

        # Candidate activation
        X_h = slice_(Xi, 'h')
        c_preact = X_h + layer_normalize(T.dot(last_h, Uh), self.g_uh, self.b_uh) * gate_r
        c = self.activation_fct(c_preact)
        h = (1 - gate_z) * last_h + gate_z * c

        return h
