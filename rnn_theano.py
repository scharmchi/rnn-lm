import numpy as np
import operator
import theano
import theano.tensor as T
import utils_rnn


class RNNTheano:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # randomly initialize
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

        # create Theano shared variable
        self.U = theano.shared(name="U", value=U.astype(theano.config.floatX))
        self.V = theano.shared(name="V", value=V.astype(theano.config.floatX))
        self.W = theano.shared(name="W", value=W.astype(theano.config.floatX))
        # store theano graph here
        self.theano_graph = {}
        self.__theano_build__()

    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')

        def fprop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:, x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o, s], updates = theano.scan(
            fprop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)

        self.fprop = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        # SGD
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function([x, y, learning_rate], [],
                                        updates=[(self.U, self.U - learning_rate * dU),
                                        (self.V, self.V - learning_rate * dV),
                                        (self.W, self.W - learning_rate * dW)])

    def calculate_avg_loss(self, X, Y):
        loss = np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])
        # divide by total number N to get the avg
        num_words = np.sum((len(y_i) for y_i in Y))
        loss /= num_words
        return loss

    def grad_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_grad = self.bptt(x, y)
        model_params = ['U', 'V', 'W']
        for param_index, param_name in enumerate(model_params):
            # get actual param by name!! WOW.. WTF!!
            param = operator.attrgetter(param_name)(self)
            print("Performing gradient check for parameter %s with size %d." % (param_name, param.shape))
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index

        pass

