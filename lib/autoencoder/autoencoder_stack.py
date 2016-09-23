#!/usr/bin/env python3

r"""
.. module:: autoencoder_blocks
   :platform: Linux
   :synopsis: Implementation of Convolutional Autoencoder Stack

Contains the implementation of the complete stack of the convolutional autoencoder

.. :moduleauthor: Matteo Ragni
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from autoencoder_helpers import ArgumentError
from autoencoder_settings import ConvAutoEncSettings
from autoencoder_blocks import ConvAutoEnc

#     _       _                       _           ___ _           _
#    /_\ _  _| |_ ___ _ _  __ ___  __| |___ _ _  / __| |_ __ _ __| |__
#   / _ \ || |  _/ -_) ' \/ _/ _ \/ _` / -_) '_| \__ \  _/ _` / _| / /
#  /_/ \_\_,_|\__\___|_||_\__\___/\__,_\___|_|   |___/\__\__,_\__|_\_\
class ConvAutoEncStack:
    r"""
    The **creation of a stack** is different with respect to the learning of a single layer.

    Some attribute of the class:

     * :py:attr:`~caes` is a list of convolutional autoencoder blocks, from the outermost to the innermost
     * :py:attr:`~inception` is a placeholder for the hallucination input
     * :py:attr:`~h_enc` output of encoder
     * :py:attr:`~error` is the error of the outermost layer
     * :py:attr:`~graph` is the graph on which the model is defined
     * :py:attr:`~session` is the session on which the optimizers are defined
     * :py:attr:`~learning_rate` is an optimization hyperparameter
     * :py:attr:`~single_optimizer` define if optimize one shoot or one layer at time

    """

    def __init__(self, config_tuple, learning_rate, single_opt=True):
        r"""
        Initialize a new series of connected block. The configuration of the single block
        is specified with a :class:`ConvAutoEncSettings`. The block order (from the exterior to the interior)
        is defined through a tuple of settings

        Other operations are:

         * initialization of a new graph for the model
         * initialization of an interactive session
         * definition of input, decoder, encoder, cost and particular optimizer for each block

        :param config_tuple: the series of configurations
        :type config_tuple: tuple
        :param learning_rate: learning rate decay hyperparameter
        :type learning_rate: float
        :param single_opt: type of optimization procedure
        :type single_opt: bool
        :returns: :class:`ConvAutoEncStack`
        :raises: AssertionError
        """
        assert type(config_tuple) is tuple, "Required a tuple of ConvAutoEncSettings"
        assert type(learning_rate) is float, "Learning rate must be a float"
        assert type(single_opt) is bool, "Optimization procedure must be a bool"
        self.caes = []
        self.learning_rate = learning_rate
        self.single_optimizer = single_opt

        for c in config_tuple:
            assert type(c) is ConvAutoEncSettings, "Element in config tuple is not ConvAutoEncSettings"
            self.caes.append(ConvAutoEnc(c, False))

        self.initializeGraph()
        self.initializeSession()
        self.initializeBlocks()
        self.defineSaver()

    def len(self):
        r"""
        Returns the number of block inside the stack

        :returns: int
        """
        return len(self.caes)

    def initializeGraph(self, g=None):
        r"""
        Define a new graph or assign a graph to all blocks

        :param g: the graph or ``None`` to create a new one
        :type g: tensorflow.Graph
        :returns: :class:`ConvAutoEncStack` current instance
        :raises: AssertionError
        """
        if g:
            assert type(g) is tf.Graph, "g must be a tensorflow.Graph"
            self.graph = g
        self.graph = tf.Graph()
        for cae in self.caes:
            cae.graph = self.graph
        return self

    def initializeSession(self):
        r"""
        Initialize an interactive session that will be used inside the stack. The session inherit the graph
        contained in the property :py:attr:`graph`

        :returns: tensorflow.InteractiveSession
        """
        self.session = tf.InteractiveSession(graph=self.graph)
        return self.session

    def close(self):
        r"""
        Closes the interactive session

        :returns: :class:`ConvAutoEncStack` current instance
        """
        self.session.close()
        return self

    def initializeBlocks(self):
        r"""
        The idea of this function is to create the structure of our model. To understand is
        better to use an example. Let's say that we have a model with 3 blocks.

        .. image:: _static/stack.png
             :align: center
             :alt: Representation of the example

        The idea of this function is to create first:
        $$
        g(x) = E_3(E_2(E_1(x)))
        $$
        and then build
        $$
        y = D_1(D_2(D_3(f(x)))) = g(h(x))
        $$
        it is not so easy to build this kind of model, and I'm not sure this is done correctly.
        All the variables seem correct during debugging. We will see in training tests.

        The training is another problem by itself, because I will need to detach temporarily section
        of the graph, and for now I don't know if it is possible. Let's say we want to train the layer
        number 2, we need to short circuit the model in suche a way that:
        $$
        h_2(x) = E_2(E_1(x))
        $$
        and
        $$
        y_2 = D_1(D_2(h_2(x)))
        $$
        and optimize between \\(x\\) and \\(y\\), withouth changing external variables (e.g.: the first block variables)...

        A new added feature is the cumulative error. if :py:attr:`single_optimizer` is set to ``True``, than
        the error that will be considered by each block is the cumulative error of the whole structure

        $$
        \\mathrm{min} \\left( \\sum_{c \\, \\in \\, \\mathrm{CAEs}} (y_c - x_c)^2 \\right)
        $$

        :warning: I know that this function sometimes raises a :py:attr:`TypeError` that is ignored by someone, and I found no way why this should happen...

        :returns: :class:`ConvAutoEncStack` current instance
        """
        # Defines encoding layer
        inps = None
        for cae in self.caes[0:self.len() - 1]:
            cae.defineInput(inps)
            cae.defineEncoder(False)
            inps = cae.h_enc
        self.h_enc = self.caes[self.len() - 1].defineInput(inps).defineEncoder().defineDecoder().defineCost().y

        # Defines decoding layer
        # TODO From this point on the function raises an exception TypeError. Why?
        outs = self.h_enc
        for cae in reversed(self.caes[0:self.len() - 1]):
            cae.h_dec = outs
            cae.defineDecoder()
            cae.defineCost()
            outs = cae.y

        # Define hallucination layer
        self.inception = tf.placeholder(tf.float32, tuple(self.h_enc.get_shape().as_list()), name="hallucinate-input")
        outs = self.inception
        for cae in reversed(self.caes[0:self.len() - 1]):
            cae.defineHallucination(outs)
            outs = cae.hallucinated

        # Defines the optimization target
        target = None
        if self.single_optimizer:
            target = self.caes[0].error
            for cae in self.caes[1:self.len()]:
                target += cae.error
            with tf.name_scope("summaries"):
                tf.scalar_summary("cumulative-cost", target)
        else:
            target = self.caes[0].error

        # Define the optimizator
        for cae in self.caes:
            cae.defineOptimizer(target=target)

        # FIXME : Error function redefined
        # self.error = self.caes[0].error
        self.error = target

        self.singleOptimizer()

        self.session.run(tf.initialize_all_variables())
        return self

    def trainBlocks(self):
        r"""
        This function should actually perform the rendering exposing an already short circuited block
        (look at :func:`initializeBlocks` for more information about).

        There is a ``yield`` context that exposes:

         * The current session
         * the level of the currently trained block
         * the currently trained block
         * the very imput layer :py:attr:`x` of the external block

        so it can be used as follows::

            >>> for session, n, cae, x in stack.trainBlocks():
            ...     print("Training block {}".format(n))
            ...     session.run(cae.optimizer, feed_dict={x: dataset})
            Training block 0
            Training block 1
            # and so on...

        The training iterates through the model **reconnecting temporarily** the encoder output and
        the decoder input.

        :returns: :class:`ConvAutoEncStack` current instance
        """
        for n in range(0, self.len()):
            cae = self.caes[n]
            temp_h = cae.h_dec
            cae.h_dec = cae.h_enc
            yield self.session, n, cae, self.caes[0].x
            cae.h_dec = temp_h
        return self

    def writeConfiguration(self, f):
        r"""
        Writes the caes configuration in an ascii file.

        :param f: filename to write into
        :type f: str
        :returns: :class:`ConvAutoEncStack` current instance
        """
        assert type(f) is str, "filename must be a str"
        with open(f, "w") as fp:
            for cae in self.caes:
                fp .write(str(cae.settings))
                fp.write("\n")
        return self

    def singleOptimizer(self):
        r"""
        Whole elements optimizer. Optimize everything in only one round, using the external error

        :returns: :class:`ConvAutoEncStack` current instance
        """
        with tf.name_scope("common-optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate,
                name="common-optimizer").minimize(self.error)
        return self

    def defineSaver(self):
        r"""
        Define a ``tf.train.Saver`` object to save variables of the autoencoder.
        For each block and layer, it will save:

         * weights
         * biases for encoder
         * biases for decoder

        :returns: :class:`ConvAutoEncStack` current instance
        """
        var_list = []
        for cae in self.caes:
            for layer in range(cae.layers):
                var_list.append(cae.weights[layer])
                var_list.append(cae.biases_encoder[layer])
                var_list.append(cae.biases_decoder[layer])
        self.saver = tf.train.Saver(var_list)
        return self

    def save(self, filename, session=None):
        r"""
        Saves current graph variables value.

        :param filename: check point file name
        :type filename: str
        :param session: current active session, defaults to ``self.session``
        :type session: ``tf.Session``
        :raises: AssertionError
        """
        assert type(filename) is str, "filename must be a str"
        # assert type(session) is tf.Session, "session must be a tf.Session"
        if session is None:
            session = self.session
        self.saver.save(session, filename)
        return self

    def restore(self, filename, session=None):
        r"""
        Restores a previous session. It will provide only:

         * weights
         * biases for encoder
         * biases for decoder

        :param filename: check point file name
        :type filename: str
        :param session: current active session, defaults to ``tf.get_default_session()``
        :type session: ``tf.Session``
        :raises: AssertionError
        """
        assert type(filename) is str, "filename must be a str"
        # assert type(session) is tf.Session, "session must be a tf.Session"
        if session is None:
            session = self.session
        self.saver.restore(session, filename)
        return self
