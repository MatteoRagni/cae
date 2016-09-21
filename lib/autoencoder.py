#!/usr/bin/env python3

r"""
.. module:: autoencoder
   :platform: Linux
   :synopsis: Core of the convolutional autoencoder

This module implements a **Convolutional Autoencoders**. There are four classes implemented inside
this module. An :class:`ArgumentError` exceptions class and:

 * :class:`ConvAutoEncSettings`: an helper class for the settings of a single convolutional autoencoder layer.
   All properties of this class are exposed into the convolutional layer class.
 * :class:`ConvAutoEnc`: is the simpler layer that is possible to define, and is the layer that is actually
   trained by itself. The settings of the layers inside this layer (e.g. stride, padding and weight size)
   atre shared and defined in the helper class.
 * :class:`CombinedAutoencoder`: is the class that contains all the simpler layer and is responsible for the
   learning process of the single layers, one at the time.

It is possible to use GPUs to train this model (and actually it is really faster if used). There are two
environment variables that can be interesting for multi-GPU systems and for commercial GPUs in general:

* **``CUDA_VISIBLE_DEVICES``**: define which GPU will be used as n.0 (``:/gpu0``)
* **``TF_MIN_GPU_MULTIPROCESSOR_COUNT``**: define the minimum number of multiprocessor to allow the use
  of a gpu (for me is 2).

.. :moduleauthor: Matteo Ragni
"""

# Importing libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
import tensorflow as tf
# import ipdb


class ArgumentError(ValueError):
    """
    Helper class to make easier to understand when an
    argument contains an error. Inherit everithing from
    the :class:`ValueError` class
    """
    pass

#  ___      _   _   _                 ___ _
# / __| ___| |_| |_(_)_ _  __ _ ___  / __| |__ _ ______
# \__ \/ -_)  _|  _| | ' \/ _` (_-< | (__| / _` (_-<_-<
# |___/\___|\__|\__|_|_||_\__, /__/  \___|_\__,_/__/__/
#                         |___/


class ConvAutoEncSettings(object):
    r"""
    Helper class to better define the simple layer configurations for the convolutional autoencoder.
    This class handles all the checks on the input, and exposes some properties to the :class:`ConvAutoEnc`
    object to simplify the configuration even before building the graph.
    """
    def __init__(self):
        r"""
        Initialize all the properties to `None`. Everything must be initialized or, when **called by
        the layer it will raise an error**

        :returns: :class:`ConvAutoEncSettings` new instance
        """
        self.__input_shape       = None
        self.__corruption        = None
        self.__corruption_min    = None  # 0
        self.__corruption_max    = None  # 0
        self.__depth_increasing  = None
        self.__layers            = None
        self.__patch_size        = None
        self.__strides           = None
        self.__padding           = None  # SAME / VALID
        self.__residual_learning = None
        self.__prefix_name       = None

    # Check methods
    def checkComplete(self):
        r"""
        Check completeness of the settings. **Raise an error if everything is not explicitly set**.

        :returns: :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError
        """
        if not(self.input_shape       is not None and
               self.corruption        is not None and
               self.layers            is not None and
               self.patch_size        is not None and
               self.strides           is not None and
               self.padding           is not None and
               self.residual_learning is not None and
               self.prefix_name       is not None and
               self.depth_increasing is not None):
            self._inspect()
            raise ArgumentError("Some of the element are not defined")
        if self.corruption is True:
            if not(self.corruption_min is not None and self.corruption_max is not None):
                self._inspect()
                raise ArgumentError(
                    "You request corruption limits but corruption is not defined")
        return self

    @property
    def input_shape(self):
        r"""
        **Property getter:**

        This property defines the shape of the input layer tensor in the form of ("NHWC"):

         1. examples batch size
         2. images height
         3. image width
         4. image depth (usually 1 for monochrome images, 3 for RGB images and 4 for images with alpha)

        As for now image width and image height must be equal, since the convolutional autoencoder is
        defined with the constraint on weights:
        \\[
        W_{\\mathrm{out}} = W^{T}_{\\mathrm{in}}
        \\]

        :returns: list -- the input shape

        **Property setter**:

        Requires a ``list`` of positive ``int`` greater than zero as input.
        **Please note that image width and image height must be equal**.
        Th first element of the list, that is the batch size, can be ``None``.

        :param value: the input shape
        :type value: list
        :returns: :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, value):
        if value:
            self._checkList(value)
            for v in value[1:-1]:
                self._checkInt(v)
                if v <= 0:
                    raise ArgumentError("Values must be positive")
            assert value[1] == value[2], "Image width and image height must be equal"
        self.__input_shape = value
        return self

    @property
    def corruption(self):
        r"""
        **Property getter**

        Refers to corruption of the input input. It is a boolean value. Corruption will create
        an effect very similar to the dropout in deep network.

        :returns: bool -- corruption settings

        **Property setter**

        It requires a boolean value. If ``False``,
        automatically sets ``corruption_min`` and ``corruption_max`` to 0.

        :param value: corruption setting
        :type value: bool
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__corruption

    @corruption.setter
    def corruption(self, value):
        if value:
            self._checkBool(value)
        if value is False:
            self.corruption_min = self.corruption_max = 0
        self.__corruption = value
        return self

    @property
    def corruption_min(self):
        r"""
        **Property getter**

        Specify corruption noise minimum value. Corruption is centered in
        the mean of corruption min and max.

        :returns: float -- corruption minimum value

        **Property setter**

        Please be aware that this value
        must be smaller than ``corruption_max``, if set, or an error will be raised. It must be a float.

        :param value: corruption setting, minimum value
        :type value: float
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__corruption_min

    @corruption_min.setter
    def corruption_min(self, value):
        if value:
            self._checkFloat(value)
        if self.corruption_max:
            if value > self.corruption_max:
                raise ArgumentError("corruption_min ({}) is greater than corruption_max ({})".format(
                    self.corruption_min, self.corruption_max))
        self.__corruption_min = value
        return self

    @property
    def corruption_max(self):
        r"""
        **Property getter**

        Specify corruption maximum value. Corruption is centered in
        the mean of corruption min and max.

        :returns: float -- corruption maximum value

        **Property setter**

        Please be aware that this value
        must be greater than ``corruption_min``, if set, or an error will be raised. It must be a float.

        :param value: corruption setting, maximum value
        :type value: float
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__corruption_max

    @corruption_max.setter
    def corruption_max(self, value):
        if value:
            self._checkFloat(value)
        if self.corruption_min:
            if value < self.corruption_min:
                raise ArgumentError("corruption_min ({}) is greater than corruption_max ({})".format(
                    self.corruption_min, self.corruption_max))
        self.__corruption_max = value
        return self

    @property
    def layers(self):
        r"""
        **Property getter**

        Define the number of the layer in this unit of the autoencoder.

        :returns: int -- number of layers in this block

        **Property setter**

        Specifies the number of layers in this unit of the convolutional autoencoder.
        Must be an integer greater or equal to one.

        :param value: number of layers in this block
        :type value: int
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__layers

    @layers.setter
    def layers(self, value):
        if value:
            self._checkInt(value)
            if value < 1:
                raise ArgumentError("Number of layers must be positive")
        self.__layers = value
        return self

    @property
    def patch_size(self):
        r"""
        **Property getter**

        Size of the patch. Weights tensor size is (convention "NHWC"):
        $$
        \\mathrm{dim}(W) = \\mathrm{input\\_depth}\\times\\mathrm{patch\\_size}^2\\times\\mathrm{input\\_depth}
        $$

        :returns: int -- patch size

        **Property setter**

        Sets the convolutional weigths tensor patch (the two inner dimensions). Must
        be an ``int`` greater than one.

        :param value: patch size
        :type value: int
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__patch_size

    @patch_size.setter
    def patch_size(self, value):
        if value:
            self._checkInt(value)
            if value <= 1:
                raise ArgumentError("Values must be positive, greater than one")
        self.__patch_size = value
        return self

    @property
    def depth_increasing(self):
        r"""
        **Property getter**

        Returns the increasing depth. Final depth of the filter will be:
        $$
        \\mathrm{output\\_size} = \\mathrm{input\\_size} + \\mathrm{depth\\_increasing}
        $$

        :returns: int -- the increasing in depth of the block

        **Property setter**

        Requires an `int` positive or 0.

        :param value: patch size
        :type value: int
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__depth_increasing

    @depth_increasing.setter
    def depth_increasing(self, value):
        if value:
            self._checkInt(value)
            if value < 0:
                raise ArgumentError("Values must be positive or zero")
        self.__depth_increasing = value
        return self

    @property
    def strides(self):
        r"""
        **Property getter**

        Returns the strides of the convolutional layer. The strides dimensions are define
        the reduction in size of the image:
        $$
        \\mathrm{dim}(S) = 1 \\times \\mathrm{red}_{x} \\times \\mathrm{red}_{y} \\times 1
        $$

        The first 1 works on batch elements number, while the last works on depth. With 1, no input
        is skipped.

        :returns: list -- strides

        **Property setter**

        It should be a list with four positive ``int``,
        greater than 0

        :param value: strides
        :type value: list
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__strides

    @strides.setter
    def strides(self, value):
        if value:
            self._checkList(value, 4)
            for v in value:
                self._checkInt(v)
                if v <= 0:
                    raise ArgumentError("Value must be positive")
        self.__strides = value

    @property
    def padding(self):
        """
        **Property getter**

        The padding configuration is a string (``SAME`` or ``VALID``)

        :returns: str - padding configuration

        **Property setter**

        The padding property is a string in the form ``SAME`` or ``VALID``:

         * ``SAME``: Round up (partial windows are included)
         * ``VALID``: Round down (only full size windows are considered).

        :param value: padding string
        :type value: str
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__padding

    @padding.setter
    def padding(self, value):
        if value:
            self._checkStr(value)
            if value != "SAME" and value != "VALID":
                raise ArgumentError(
                    "Padding can be SAME or VALID. Received {}".format(value))
        self.__padding = value
        return self

    @property
    def prefix_name(self):
        """
        **Property getter**

        Prefix name for this particular layer. It will be usefull to identify it inside
        the TensorBoard visualization tool (will be used in front of all ``name`` property)

        :returns: str -- actual prefix name used

        **Property setter**

        Sets the prefix name for this block. It must be a string, that will be inherit
        by the whole part of the tree that represents this block.

        :param value: prefix name
        :type value: str
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__prefix_name

    @prefix_name.setter
    def prefix_name(self, value):
        if value:
            self._checkStr(value)
        self.__prefix_name = value
        return self

    @property
    def residual_learning(self):
        r"""
        **Property getter**

        Defines the nature of the learning. If this flag is ``False``, the objective funtion in
        $$
        \\mathrm{min}\\,(y - x)^2 = \\mathrm{min}\\,(D(E(x)) - x)^2
        $$
        while, if ``True``, it will try to learn:
        $$
        \\mathrm{min}\\,(D(E(x)) + x - x)^2 = \\mathrm{min}\\,(D(E(x)))^2
        $$
        (thus, the residuals only).

        **Property setter**

        Define if learning the residuals: requires a ``bool`` ad input. If true, the learning
        will be on the residuals.

        :param value: activate residuals learning
        :type value: bool
        :returns: ConvAutoEncSettings -- actual :class:`ConvAutoEncSettings` instance
        :raises: ArgumentError, AssertionError
        """
        return self.__residual_learning

    @residual_learning.setter
    def residual_learning(self, value):
        if value:
            self._checkBool(value)
        self.__residual_learning = value
        return self

    # Other methods
    def _checkType(self, obj, tp):
        r"""
        Helper function, check with an :func:`assert` the type specified.

        :param obj: object to be tested
        :param tp: type to be checked against
        :returns: bool
        :raises: AssertionError
        """
        assert type(obj) is tp, "%r is not of the correct type %r" % (obj, tp)
        return True

    def _checkStr(self, obj):
        r"""
        ``str`` check helper function

        :param obj: object to be tested
        :returns: bool
        :raises: AssertionError
        """
        return self._checkType(obj, str)

    def _checkInt(self, obj):
        r"""
        ``int`` check helper function

        :param obj: object to be tested
        :returns: bool
        :raises: AssertionError
        """
        return self._checkType(obj, int)

    def _checkFloat(self, obj):
        r"""
        ``float`` check helper function

        :param obj: object to be tested
        :returns: bool
        :raises: AssertionError
        """
        return self._checkType(obj, float)

    def _checkBool(self, obj):
        r"""
        ``bool`` check helper function

        :param obj: object to be tested
        :returns: bool
        :raises: AssertionError
        """
        return self._checkType(obj, bool)

    def _checkList(self, obj, size=0):
        r"""
        ``list`` check helper function. If specified check also size.

        :param obj: object to be tested
        :param size: size of the list
        :type size: int
        :returns: bool
        :raises: AssertionError
        """
        self._checkType(obj, list)
        self._checkInt(size)
        if size > 0:
            assert len(obj) == size, "Size error for list: %d != %d" % (
                len(obj), size)
        return True

    def _checkHash(self, obj, keylist=None):
        r"""
        ``dict`` check helper function. If specified a list of keys specifies also the presence
        of all the keys in the dictionary.

        :param obj: object to be tested
        :param keylist: list of keys that must be in the `dict`
        :type keylist: list
        :returns: bool
        :raises: AssertionError
        """
        self._checkType(obj, dict)
        if keylist is not None:
            self._checkList(keylist)
            for k in keylist:
                try:
                    obj[k]
                except KeyError:
                    raise KeyError(
                        "Key %r is not defined in object %r" % (k, obj))
        return True

    def _inspect(self):
        """Inspect function. Print :func:`__str__` on screen"""
        print(self.__str__())

    def __str__(self):
        """
        Convert object into ``str``

        :returns: str -- the string that represents the object
        """
        return "  Convolutional Autoencoder Settings"                    + "\n" + \
               "--------------------------------------"                  + "\n" + \
               " - prefix_name       = {}".format(self.prefix_name)      + "\n" + \
               " - input_shape       = {}".format(self.input_shape)      + "\n" + \
               " - corruption        = {}".format(self.corruption)       + "\n" + \
               " - corruption_min    = {}".format(self.corruption_min)   + "\n" + \
               " - corruption_max    = {}".format(self.corruption_max)   + "\n" + \
               " - layers            = {}".format(self.layers)           + "\n" + \
               " - depth_increasing  = {}".format(self.depth_increasing) + "\n" + \
               " - patch_size        = {}".format(self.patch_size)       + "\n" + \
               " - strides           = {}".format(self.strides)          + "\n" + \
               " - padding           = {}".format(self.padding)          + "\n" + \
               " - residual learning = {}".format(self.residual_learning)


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
            self.defineSaver()

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
     * :py:attr:`~FLAGS` is a copy of :py:attr:`tensorflow.flags.FLAGS`

    """

    def __init__(self, config_tuple):
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
        :returns: :class:`ConvAutoEncStack`
        :raises: AssertionError
        """
        assert type(config_tuple) is tuple, "Required a tuple of ConvAutoEncSettings"
        self.caes = []

        self.FLAGS = tf.app.flags.FLAGS

        for c in config_tuple:
            assert type(c) is ConvAutoEncSettings, "Element in config tuple is not ConvAutoEncSettings"
            self.caes.append(ConvAutoEnc(c, False))

        self.initializeGraph()
        self.initializeSession()
        self.initializeBlocks()

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

        A new added feature is the cumulative error. if ``tensorflow.flags.FLAGS.cumulative_error`` is set to ``True``, than
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
        if self.FLAGS.cumulate_error:
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
                self.FLAGS.learning_rate,
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
            for layer in self.layers:
                var_list.append(cae.weights[layer])
                var_list.append(cae.biases_encoder[layer])
                var_list.append(cae.biases_decoder[layer])
        self.saver = tf.train.Saver(var_list)
        return self

    def save(self, filename, session=tf.get_default_session()):
        r"""
        Saves current graph variables value.

        :param filename: check point file name
        :type filename: str
        :param session: current active session, defaults to ``tf.get_default_session()``
        :type session: ``tf.Session``
        :raises: AssertionError
        """
        assert type(filename) is str, "filename must be a str"
        assert type(session) is tf.Session, "session must be a tf.Session"
        self.saver.save(session, filename)
        return self

    def restore(self, filename, session=tf.get_default_session()):
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
        assert type(session) is tf.Session, "session must be a tf.Session"
        self.saver.restore(session, filename)
        return self
