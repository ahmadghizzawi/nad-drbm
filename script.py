import cPickle
import gzip
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder
import theano
import theano.tensor as T

from utils import load_data

"""Implementation of the discriminative RBM trained with stochastic gradient
descent in Theano.
"""

class DiscriminativeRBM(object):
    """Discriminative restricted Boltzmann machine class

    The discriminative RBM is fully characterized by four parameters:
      * Weight matrix W between input units in the visible layer and the hidden
        units.
      * Weight matrix U between output units in the visible layer and the
        hidden units.
      * Bias vector c of the hidden layer units.
      * Bias vector d of the output units.

    Classification is done using the class-wise posterior probabilities
    predicted by the model in its output units.
    """

    def __init__(self, input, n_inputs, n_classes, n_hiddens,
                 learning_rate=1e-2, weight_decay=1e-4, rng=0):
        """ Initialize the parameters of the DRBM

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
            architecture (one minibatch)

        :type n_inputs: int
        :param n_inputs: number of input units, the dimension of the space in
            which the datapoints lie

        :type n_classes: int
        :param n_outputs: number of output units, the dimension of the space in
            which the labels lie

        :type n_hiddens: int
        :param n_hiddens: number of hidden units, the dimension of the
            latent-variable space

        :learning_rate: float
        :param learning_rate: learning rate for stochastic gradient descent

        :weight_decay: float
        :param weight_decay: weight decay regularisation for stochastic
            gradient descent
        """
        # Model parameters
        self.n_inputs=n_inputs
        self.n_classes=n_classes
        self.n_hiddens=n_hiddens
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.rng=rng

        # Class-hidden weights
        # U_init = np.asarray(rng.rand(n_classes, n_hiddens) * 1e-3,
        # dtype=theano.config.floatX)
        U_init = np.asarray((rng.rand(n_classes, n_hiddens) * 2 - 1) / \
                np.sqrt(max(n_classes, n_hiddens)), dtype=theano.config.floatX)
        self.U = theano.shared(U_init, name='U')

        # Input-hidden weights
        # W_init = np.asarray(rng.rand(n_inputs, n_hiddens) * 1e-3,
#                            dtype=theano.config.floatX)
        W_init = np.asarray((rng.rand(n_inputs, n_hiddens) * 2 - 1) / \
                np.sqrt(max(n_inputs, n_hiddens)), dtype=theano.config.floatX)
        self.W = theano.shared(W_init, name='W')

        # Hidden biases
        c_init = np.zeros((n_hiddens,), dtype=theano.config.floatX)
        self.c = theano.shared(c_init, name='c')

        # Class biases
        d_init = np.zeros((n_classes,), dtype=theano.config.floatX)
        self.d = theano.shared(d_init, name='d')

        # Posterior probability
        self.p_y_given_x = self.p_y_given_x_fn(input)

        # Predict class labels
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # L2-regularizer
        self.L2 = (self.W**2).sum()

        # Parameters of the model
        self.params = [self.U, self.W, self.c, self.d]


    def p_y_given_x_fn(self, X):
        """Return the posterior probability of classes given inputs and model
        parameters (p(y|x, theta)).

        Input
        -----
        :type X: theano.shared
        :param X: Input data matrix

        Output
        ------
        :type p: ???
        :param p: Posterior class probabilities of the input data points as
            given by the model
        """
        # NOTE: Doing this for now as theano.scan returns an error when passing
        # class variables directly into it.
        U = self.U
        W = self.W
        c = self.c
        d = self.d
        Y_class = theano.shared(np.eye(self.n_classes,
                                          dtype=theano.config.floatX),
                                name='Y_class')

        # Compute hidden states
        s_hid = theano.tensor.dot(X, W) + c

        # Compute energies
        energies, updates = theano.scan(lambda y_class, U, s_hid:
                                        s_hid + theano.tensor.dot(y_class, U),
                                        sequences=[Y_class],
                                        non_sequences=[U, s_hid])

        # Compute log-posteriors
        log_p, updates = theano.scan(
                lambda d_i, e_i: d_i + \
                        theano.tensor.sum(theano.tensor.log(1+theano.tensor.exp(e_i)),
                                          axis=1),
                 sequences=[d, energies], non_sequences=[])
        log_p = log_p.T # TODO: See if this can be avoided

        # Compute unnormalized posteriors
        unnorm_p = theano.tensor.exp(log_p - theano.tensor.max(log_p, axis=1,
                                                               keepdims=True))

        # Compute posteriors
        p = unnorm_p / theano.tensor.sum(unnorm_p, axis=1, keepdims=True)

        return p


    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def sgd_optimization(n_hiddens=50, learning_rate=0.13, weight_decay=1e-4,
                     n_epochs=1000, dataset='kddcup.data_10_percent.gz', test_dataset='corrected.gz', batch_size=600):
    """Demonstrate stochastic gradient descent optimization of a Discriminative
    restricted Boltzmann machine.

    The default dataset for the demonstration is MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type weight_decay: float
    :param weight_decay: weight-decay for regularization

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the dataset file in the same format as the
                    MNIST dataset available at
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    :type batch_size: int
    :param batch_size: size of data-batches used for training
    """
    #############
    # LOAD DATA #
    #############
    datasets = load_data(dataset, test_dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    n_inputs = valid_set_x.get_value().shape[1]
    n_classes = np.shape(np.unique(np.concatenate((train_set_y.eval(),
                                                   test_set_y.eval(),
                                                   valid_set_y.eval()),
                                                   axis=0)))[0]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    rng = np.random.RandomState(666)

    # construct the DRBM class
    classifier = DiscriminativeRBM(input=x, n_inputs=n_inputs,
                                   n_classes=n_classes, n_hiddens=n_hiddens,
                                   weight_decay=weight_decay, rng=rng)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y) + \
            classifier.weight_decay * classifier.L2

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_U = T.grad(cost=cost, wrt=classifier.U)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_c = T.grad(cost=cost, wrt=classifier.c)
    g_d = T.grad(cost=cost, wrt=classifier.d)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.U, classifier.U - learning_rate * g_U),
               (classifier.W, classifier.W - learning_rate * g_W),
               (classifier.c, classifier.c - learning_rate * g_c),
               (classifier.d, classifier.d - learning_rate * g_d)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization(n_hiddens=500, learning_rate=0.1, n_epochs=5)
    sgd_optimization(n_hiddens=500, learning_rate=0.1, n_epochs=10)
    sgd_optimization(n_hiddens=500, learning_rate=0.1, n_epochs=15)
