#!/usr/bin/env python3

r"""
.. module:: autoencoder_blocks
   :platform: Linux
   :synopsis: Implementation of Convolutional Autoencoder Block

Contains the implementation of the single of the convolutional autoencoder

.. :moduleauthor: Matteo Ragni
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from autoencoder_helpers import ArgumentError
from autoencoder_settings import ConvAutoEncSettings

#   ___                 _      _   _              _       _                            _
#  / __|___ _ ___ _____| |_  _| |_(_)___ _ _     /_\ _  _| |_ ___  ___ _ _  __ ___  __| |___ _ _
# | (__/ _ \ ' \ V / _ \ | || |  _| / _ \ ' \   / _ \ || |  _/ _ \/ -_) ' \/ _/ _ \/ _` / -_) '_|
#  \___\___/_||_\_/\___/_|\_,_|\__|_\___/_||_| /_/ \_\_,_|\__\___/\___|_||_\__\___/\__,_\___|_|
class ConvAutoEnc(object):
    r"""
    This class implements a simple **convolutional autoencoder** block. This block may contains different
    layers, and it is the the unit that is optimized. One single block is defined by an encoder mapping
    \\(E(\cdot)\\) and a decoder mapping \\(D(\cdot)\\).

    In particular:
    $$
    h_{E} = E(x)
    $$
    and
    $$
    y = D(h_{D})
    $$
    In optimization (learning), since we are only optimizing one single block at the time,
    $$
    h_{E} = h_{D}
    $$
    whilst, other blocks can be injected inside, thus leading to \\(h_{E} \\neq h_{D}\\).

    If more \\(n\\) layers are requested, there will be multiple layers that will share the same configuration
    in the form:
    $$
    E(\\cdot) = (E_{n} \\circ \\dots \\circ E_{1})(\\cdot)
    $$
    and decoder in the form:
    $$
    D(\\cdot) = (D_{n} \\circ \\dots \\circ D_{1})(\\cdot)
    $$
    For what concerns the single layers functions we have:
    $$
    E_{i}(\\cdot) = \\mathrm{LeakRelu}\\left( \\mathrm{Conv}(W_{i}, \\cdot) + b_{E,i} \\right)
    $$
    whilst the decoder is in the form:
    $$
    D_{i}(\\cdot) = \\mathrm{LeakRelu}\\left( \\mathrm{Deconv}(W^{T}_{i}, \\cdot) + b_{D,i} \\right)
    $$
    (the weights in the two operations are shared, in trasposed version). Interfaces of the block are:

     * :py:attr:`~x`: input placeholder
     * :py:attr:`~y`: output op
     * :py:attr:`~h_enc`: innest layer op
     * :py:attr:`~h_dec`: decoder input (it should be an op)
     * :py:attr:`~weights`: a list of variables (weights for each encoding level)
     * :py:attr:`~biases_encoder`: a list of biases variable for the encoder
     * :py:attr:`~biases_decoder`: a list of biases for the decoder
    """

    def __init__(self, settings, build_now=True):
        r"""
        Initializes the block, using the helper class :class:`ConvAutoEncSettings`, that contains all
        about the current block configuration. The use of **GPU** is defined on the basis of the
        application :py:attr:`~FLAGS` variables.
        _Please consider that all high level attribute (withoust leading ``__``) are inherit in this
        class as first level citizen, and accessible directly, but cannot be modified after object
        instanciation.

        :param settings: settings for the current block
        :type settings: :class:`ConvAutoEncSettings`
        :param build_now: call immediately :func:`defineInput`, :func:`defineEncoder`, :func:`defineDecoder` and :func:`defineCost`
        :type build_now: bool
        :returns: :class:`CanvAutoEnc` new instance
        :raises: AssertionError
        """
        assert type(
            settings) is ConvAutoEncSettings, "Settings are not of type ConvAutoSettings"
        assert type(build_now) is bool, "build_now must be a bool"
        self.settings = settings
        self.settings.checkComplete()

        self.graph   = tf.Graph()
        self.x       = None
        self.h_enc   = None
        self.h_dec   = None
        self.y       = None
        self.weights = None
        self.error   = None

        self.biases_encoder  = None
        self.biases_decoder  = None
        self.optimizer       = None
        self.session         = None

        self.FLAGS = tf.app.flags.FLAGS

        if build_now:
            self.defineInput()
            self.defineEncoder()
            self.defineDecoder()
            self.defineCost()

    def __getattribute__(self, key):
        r"""
        Exposes all getter of the :class:`ConvAutoEncSettings` in this class
        """
        try:
            return super(ConvAutoEnc, self).__getattribute__('settings').__getattribute__(key)
        except AttributeError:
            return super(ConvAutoEnc, self).__getattribute__(key)

    def _corrupt(self, x, name="corruption"):
        r"""
        Corruption of a signal (helper function).

        The corruption is a multiplication between a ramdomly sampled tensor from an uniform
        distribution, with minimum value `corruption_min` and maximum value `corruption_max`
        The corruption acts as a simplified dropout layer.

        :param x: tensor to be corrupted
        :type x: tensorflow.Tensor, tensorflow.Variable
        :param name: a string to identify the op on the Tensorboard visualization
        :param type: str
        :returns: tensorflow.Tensor
        :raises: AssertionError
        """
        assert type(x) is tf.Tensor or type(x) is tf.Variable, "x  must be a tf.Tensor"
        assert type(name) is str, "name must be a str"
        with self.graph.as_default():
            return tf.mul(x, tf.cast(tf.random_unifor(shape=tf.shape(x),
                                                      minval=self.corruption_min,
                                                      maxval=self.corruption_max,
                                                      dtype=tf.int32)), name=self.prefix_name + "-" + name)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        r"""
        Change the current `tf.Graph` (dafault: requested on object instantiation) with another one

        :param graph: the new graph
        :type graph: tf.Graph
        :raises: AssertionError
        """
        assert type(graph) is tf.Graph
        self._graph = graph
        return self

    def leakRelu(self, x, alpha=0.2, name="leak-relu"):
        r"""
        The leaking relu is defined as follow:
        $$
        y = \\dfrac{1}{2} \\, \\left( \\alpha\\,x + (1-\\alpha)\\,x \\right)
        $$

        :param x: the input tensor of the relu module
        :type x: tensorflow.Tensor
        :param alpha: the actual \\(\\alpha\\)
        :type alpha: float
        :param name: name for the op
        :type name: str
        :returns: tensorflow.Tensor
        :raises: AssertionError
        """
        assert type(x) is tf.Tensor, "x must be a tensorflow.Tensor"
        assert type(alpha) is float, "alpha must be a float"
        assert type(name) is str, "name must be a str"
        alpha_t = tf.constant(alpha, name=(name + "-constant"))
        with self.graph.as_default():
            with tf.name_scope(name):
                return 0.5 * ((1 + alpha_t) * x + (1 - alpha_t) * abs(x))

    def defineInput(self, input=None):
        r"""
        If :py:attr:`input` is :py:attr:`None` creates a input placeholder and a corruption
        operations if required. Input is added to the summary. This is useful for the very first
        block of the autoencoder

        If :py:attr:`input` is a :py:attr:`tensorflow.Tensor` (direct or an op)

        :param input: the input shape tensor
        :type input: tensorflow.Tensor, tensorflow.Variable
        :returns: :class:`ConvAutoEnc` current instance
        :raises: AssertionError
        """
        if input is None:
            with self.graph.as_default():
                with tf.name_scope(self.prefix_name + "-input-layer"):
                    x = tf.placeholder(tf.float32, self.input_shape, name=self.prefix_name + '-x')
                    if self.corruption:
                        self.x = self._corrupt(x)
                    else:
                        self.x = x
                    self.addXsummary(self.x, self.prefix_name + '-x')
        else:
            assert type(input) is tf.Tensor or type(input) is tf.Variable, "input must be a tensorflow.Tensor"
            self.input_shape = input.get_shape().as_list()
            self.x = input
        return self

    def defineEncoder(self, connect=True):
        r"""
        Construct the encoder graph. Also assign `h_dec` as `h_enc`

        :param connect: connect directly :py:attr:`h_enc` at :py:attr:`h_dec`
        :type connect: bool
        :returns: :class:`ConvAutoEnc` current instance
        :raises: AssertionError, RuntimeError
        """
        assert type(connect) is bool, "connect must be a bool"
        if self.x is None:
            raise RuntimeError("You must define input to define the encoder")
        with self.graph.as_default():
            with tf.name_scope(self.prefix_name + "-encoder"):
                self.weights = []
                self.patches = []
                self.latent  = []
                self.shapes  = []
                self.biases_encoder  = []
                x_current    = self.x
                for layer in range(0, self.layers):
                    old_depth = x_current.get_shape().as_list()[3]
                    new_depth = old_depth + self.depth_increasing
                    patch = [self.patch_size,
                             self.patch_size, old_depth, new_depth]
                    self.shapes.append(x_current.get_shape().as_list())

                    # Naming
                    name_W    = self.prefix_name + "-weight-%d"              % layer
                    name_B    = self.prefix_name + "-enc-bias-%d"            % layer
                    name_conv = self.prefix_name + "-encoder-convolution-%d" % layer
                    name_sum  = self.prefix_name + "-encoder-sum-%d"         % layer
                    name_out  = self.prefix_name + "-encoder-out-%d"         % layer

                    W = tf.Variable(tf.truncated_normal(
                        patch, stddev=0.1), name=name_W)
                    self.addRsummary(W, name_W)
                    # self.add2summary(W, name_W)
                    B = tf.Variable(tf.zeros([new_depth]), name=name_B)
                    self.add2summary(B, name_B)

                    self.weights.append(W)
                    self.biases_encoder.append(B)

                    h_layer = tf.add(
                        tf.nn.conv2d(x_current, W, strides=self.strides,
                                     padding=self.padding, name=name_conv,
                                     use_cudnn_on_gpu=self.FLAGS.gpu_enabled),
                        B, name=name_sum
                    )
                    x_current = self.leakRelu(h_layer, name=name_out)
                self.h_enc = x_current
                if connect:
                    self.h_dec = self.h_enc
                self.addXsummary(self.h_enc, self.prefix_name + '-h_enc')
                # self.add2summary(self.h, "h")
        return self

    def defineDecoder(self):
        r"""
        Defines the decoder layer. At the end defines also `y` to reflect residual learning
        or not.

        :raises: RuntimeError
        :returns: :class:`ConvAutoEnc` current instance
        """
        if (self.x is None) or (self.h_dec is None):
            raise RuntimeError("To define a decoder you must define the encoder")
        self.biases_decoder = []
        with self.graph.as_default():
            with tf.name_scope(self.prefix_name + "-decoder"):
                x_current = self.h_dec
                for current_layer in range(0, self.layers):
                    layer       = self.layers - (current_layer + 1)
                    name_deconv = self.prefix_name + "-decoder-deconvolution-%i" % layer
                    name_sum    = self.prefix_name + "-decoder-sum-%i"           % layer
                    name_out    = self.prefix_name + "-decoder-out-%i"           % layer
                    name_B      = self.prefix_name + "-dec-bias-%d"              % layer

                    W = self.weights[layer]
                    B = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]), name=name_B)
                    self.add2summary(B, name_B)

                    self.biases_decoder.append(B)

                    shape = self.shapes[layer]
                    x_current = self.leakRelu(tf.add(
                        tf.nn.conv2d_transpose(
                            x_current, W, shape, strides=self.strides, padding=self.padding,
                            name=name_deconv),
                        B, name=name_sum
                    ), name=name_out)
                    # x_current = self.leakRelu(h_layer, name=name_out)
                if self.residual_learning:
                    self.y = x_current + self.x
                    self.addXsummary(x_current, self.prefix_name + '-y-residuals')
                else:
                    self.y = x_current
                self.addXsummary(self.y, self.prefix_name + '-y')
        self.biases_decoder = self.biases_decoder[::-1]
        return self

    def defineHallucination(self, inception):
        r"""
        Defines the hallucination layer. This particular layer is used to test
        the result of an inception from the middle layer

        :param inception: the input, that can be a placeholder or something else
        :type inception: tf.Tensor
        :raises: AssertionError, RuntimeError
        :returns: :class:`ConvAutoEnc` current instance
        """
        assert type(inception) is tf.Tensor, "Inception must be a tensorflow.Tensor"
        if self.biases_decoder is None:
            raise RuntimeError("To define an hallucination you must define the decoder")
        with self.graph.as_default():
            with tf.name_scope(self.prefix_name + "-hallucinate"):
                x_current = inception
                for current_layer in range(0, self.layers):
                    layer       = self.layers - (current_layer + 1)
                    name_deconv = self.prefix_name + "-hallucinate-deconvolution-%i" % layer
                    name_sum    = self.prefix_name + "-hallucinate-sum-%i"           % layer
                    name_out    = self.prefix_name + "-hallucinate-out-%i"           % layer

                    W = self.weights[layer]
                    B = self.biases_decoder[layer]

                    shape = self.shapes[layer]
                    # h_layer = self.leakRelu(tf.add(
                    x_current = self.leakRelu(tf.add(
                        tf.nn.conv2d_transpose(
                            x_current, W, shape, strides=self.strides, padding=self.padding,
                            name=name_deconv),
                        B, name=name_sum
                    ), name=name_out)
                    # x_current = self.leakRelu(h_layer, name=name_out)
                self.hallucinated = x_current
                self.addXsummary(self.y, self.prefix_name + '-y-hallucinate')
        return self

    def defineCost(self):
        r"""
        Define the cost function that must be otpimized for this block, as minimization
        of the sum on the square error between input and output.

        :raises: RuntimeError
        :returns: :class:`ConvAutoEnc` current instance
        """
        if (self.x is None) or (self.h_enc is None) or (self.y is None):
            raise RuntimeError("You cannot define a cost, if you not define output")
        with self.graph.as_default():
            with tf.name_scope(self.prefix_name + '-cost-function'):
                self.error = tf.reduce_sum(
                    tf.square(self.y - self.x), name=self.prefix_name + '-error-definition')
                self.add1summary(self.error, self.prefix_name + '-error')
        return self

    def defineOptimizer(self, target=None, opt="ADAM"):
        r"""
        Define the optimizer relative to this particular block. The definitions contains a list of variable
        that can be optimized by this particular block, that are:

         * :py:attr:`weights`
         * :py:attr:`biases_encoder`
         * :py:attr:`biases_decoder`

        If a target is speicified, it will be the optimizing functional. If it is ``None``, the functional will be the current
        block :py:attr:`~error`
        Please notice that a session for this particular thread must be active. If not, the function
        will raise a :class:`RuntimeError`. That means that it should be called in a :py:attr:`tensorflow.Session` context
        or after a :py:attr:`tensorflow.InteractiveSession` call. The session will be retrieved with
        :py:attr:`tensorflow.get_default_session()`. If :py:attr:`session` is not ``None``,

        As for now, only one optimizer is defined (**ADAM algorithm**), but we will se in the future
        if add some more.

        :warning: as for now, it inherits the dafault session only.

        :param target: the target for the optimization if different from layer cost
        :type target: tensorflow.Tensor, None
        :param opt: type of optimizer to define: for now only ``"ADAM"`` available
        :type opt: str
        :returns: :class:`ConvAutoEnc` current instance
        :raises: AssertionError, RuntimeError
        """
        assert type(opt) is str, "opt must be a str"
        if target is not None:
            assert type(target) is tf.Tensor, "target must be a tensor"
        else:
            target = self.error

        s = tf.get_default_session()
        if s is None:
            raise RuntimeError("Session must be declared before declaring optimizer")

        with tf.name_scope(self.prefix_name + "-" + "optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                self.FLAGS.learning_rate,
                name=self.prefix_name + "-optimizer").minimize(
                    target,
                    var_list=(self.weights + self.biases_encoder + self.biases_decoder))
        return self

    def add1summary(self, var, name):
        r"""
        Add scalar elements to the summary, using name defined

        :param var: the variable to add to the summary
        :type var: tensorflow.Tensor
        :param var: the name to be assigned in the summary
        :type name: str
        :raises: AssertionError
        """
        assert type(var) is tf.Tensor or type(var) is tf.Variable, "var must be a tf.Tensor"
        assert type(name) is str, "Name must be a string"
        with tf.name_scope("summaries"):
            tf.scalar_summary(name, var)

    def add2summary(self, var, name):
        r"""
        Add histogram elements to the summary, using name defined

        :param var: the variable to add to the summary
        :type var: tensorflow.Tensor, tensorflow.Variable
        :param name: the name to be assigned in the summary
        :type name: str
        :raises: AssertionError
        """
        assert type(var) is tf.Tensor or type(var) is tf.Variable, "var must be a tf.Tensor"
        assert type(name) is str, "Name must be a string"
        with tf.name_scope("summaries"):
            pass
            tf.histogram_summary(name, var)

    def add3summary(self, var, name, batch=1):
        r"""
        Add image elements to the summary, using name defined

        :param var: the variable to add to the summary
        :type var: tensorflow.Tensor, tensorflow.Variable
        :param name: the name to be assigned in the summary
        :type name: str
        :param batch: number of elements to sample from the batch dimension
        :type batch: int
        :raises: AssertionError
        """
        assert type(var) is tf.Tensor or type(var) is tf.Variable, "var must be a tf.Tensor"
        assert type(name) is str, "Name must be a string"
        assert type(batch) is int, "batch size must be an integer"
        assert batch > 0, "batch size must be positive"
        with tf.name_scope("summaries"):
            tf.image_summary(name, var, max_images=batch)

    def addXsummary(self, var, name, batch=1):
        r"""
        Add multi-dimensional elements to the summary, using name defined, and with `batch`
        samples from the batch dimension. The different dimension will be added as a monochromatic
        images, splitting layers by a numeric suffix.

        :param var: the variable to add to the summary
        :type var: tensorflow.Tensor, tensorflow.Variable
        :param name: the name to be assigned in the summary
        :type name: str
        :param batch: number of elements to sample from the batch dimension
        :type batch: int
        :raises: AssertionError
        """
        assert type(var) is tf.Tensor or type(var) is tf.Variable, "var must be a tf.Tensor"
        assert type(name) is str, "Name must be a string"
        assert type(batch) is int, "batch size must be a integer"
        assert batch > 0, "batch size must be positive"
        with tf.name_scope("summaries"):
            for layer in range(0, var.get_shape()[3]):
                name_l = "%s-L%d" % (name, layer)
                tf.image_summary(name_l, var[:, :, :, layer:layer + 1], max_images=batch)

    def _reshapeTensor(self, t):
        r"""
        Given a tensor of dimensions:
        $$
        \\mathrm{height} \\times \\mathrm{width} \\times \\mathrm{input\\_size} \\times \\mathrm{output\\_size}
        $$
        tries to recreate a new
        planar tensor that can be seen as an image by the summary function. That means it will be
        reshaped two times:

         * output will be concatenated on the width
         * input will be concatenated on the height

        (convention "NHWC"). The final tensor in output will have dimensions
        $$
        1 \\times \\mathrm{height} \\, \\mathrm{input\\_size} \\times \\mathrm{width} \\, \\mathrm{output\\_size} \\times 1
        $$
        This means also that tensor rank must be 4.

        :param t: tensor to be reshaped
        :type t: tensorflow.Tensor, tensorflow.Variable
        :returns: tensorflow.Tensor
        :raises: AssertionError, RuntimeError
        """
        assert type(t) is tf.Tensor or type(t) is tf.Variable, "t must be a tensorflow.Tensor"
        shape = t.get_shape().as_list()
        assert len(shape) is 4, "t rank must be 4, received %d" % len(shape)

        rows = []
        for r in range(0, shape[2]):
            cols = []
            for c in range(0, shape[3]):
                cols.append(t[:, :, r:r + 1, c:c + 1])
            rows.append(tf.concat(1, cols))
        r = tf.squeeze(tf.concat(0, rows))
        r_shape = r.get_shape().as_list()
        return tf.reshape(r, [1, r_shape[0], r_shape[1], 1])

    def addRsummary(self, var, name):
        r"""
        Uses :func:`_reshapeTensor` to add a tensor (like a weight) to the summary

        :param var: the variable to add to the summary will be squeezed
        :type var: tensorflow.Tensor, tensorflow.Variable
        :param name: the name to be assigned in the summary
        :type name: str
        :raises: AssertionError
        """
        assert type(var) is tf.Variable or type(var) is tf.Tensor, "var must be a tensorflow.Tensor"
        assert type(name) is str, "name must be a str"
        with tf.name_scope("summaries"):
            sum = self._reshapeTensor(var)
            self.add3summary(sum, name)
