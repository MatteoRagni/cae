#!/usr/bin/env python3

"""
This module implements a **Convolutional Autoencoders**. There are four classes implemented inside
this module. An `ArgumentError` exceptions class and:

 * `ConvAutoEncSettings`: an helper class for the settings of a single convolutional autoencoder layer.
   All properties of this class are exposed into the convolutional layer class.
 * `ConvAutoEnc`: is the simpler layer that is possible to define, and is the layer that is actually
   trained by itself. The settings of the layers inside this layer (e.g. stride, padding and weight size)
   atre shared and defined in the helper class.
 * `CombinedAutoencoder`: is the class that contains all the simpler layer and is responsible for the
   learning process of the single layers, one at the time.
"""

# Importing libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
import tensorflow as tf


class ArgumentError(ValueError):
    """
    Helper class to make easier to understand when an
    argument contains an error. Inherit everithing from
    the `ValueError` class
    """
    pass

#  ___      _   _   _                 ___ _
# / __| ___| |_| |_(_)_ _  __ _ ___  / __| |__ _ ______
# \__ \/ -_)  _|  _| | ' \/ _` (_-< | (__| / _` (_-<_-<
# |___/\___|\__|\__|_|_||_\__, /__/  \___|_\__,_/__/__/
#                         |___/


class ConvAutoEncSettings(object):
    """
    Helper class to better define the simple layer configurations for the convolutional autoencoder.
    This class handles all the checks on the input, and exposes some properties to the `ConvAutoEnc`
    object to simplify the configuration even before building the graph.
    """
    def __init__(self):
        """
        Initialize all the properties to `None`. Everything must be initialized or, when **called by
        the layer it will raise an error**
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
        """
        Check completeness of the settings. **Raise an error if everything is not explicitly set**.
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
        """
        Property getter for `input_shape`.

        This property defines the shape of the input layer tensor in the form of:

         1. examples batch size
         2. images width
         3. image height
         4. image depth (usually 1 for monochrome images, 3 for RGB images and 4 for images with alpha)

        As for now image width and image height must be equal, since the convolutional autoencoder is
        defined with the constraint on weights:

        $$
        W_{\mathrm{out}} = W^{T}_{\mathrm{in}}
        $$
        """
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, value):
        """ for now image width and image height must be equal, since the convolutional autoencoder is
         defined with the constraint on weights:

         $$
         W_{\mathrm{out}} = W^{T}_{\mathrm{in}}
         $$
        Property setter for `input_shape`. Requires a `list` of positive `int` greater than zero as input.
        **Please note that image width and image height must be equal**.
        """
        if value:
            self._checkList(value)
            for v in value:
                self._checkInt(v)
                if v <= 0:
                    raise ArgumentError("Values must be positive")
            assert value[1] == value[2], "Image width and image height must be equal"
        self.__input_shape = value

    @property
    def corruption(self):
        """
        Property getter for corruption (noise) into the input. It is a boolean value.
        """
        return self.__corruption

    @corruption.setter
    def corruption(self, value):
        """
        Property setter for corruption of the input. It requires a boolean value. If `False`,
        automatically sets `corruption_min` and `corruption_max` to 0.
        """
        if value:
            self._checkBool(value)
        if value is False:
            self.corruption_min = self.corruption_max = 0
        self.__corruption = value

    @property
    def corruption_min(self):
        """
        Property getter for corruption noise minimum value. Corruption is noise centered in
        the mean of corruption min and max.
        """
        return self.__corruption_min

    @corruption_min.setter
    def corruption_min(self, value):
        """
        Property setter for the corruption noise minimum value. Please be aware that this value
        must be smaller than `corruption_max`, if set, or an error will be raised. It must be a float.
        """
        if value:
            self._checkFloat(value)
        if self.corruption_max:
            if value > self.corruption_max:
                raise ArgumentError("corruption_min ({}) is greater than corruption_max ({})".format(
                    self.corruption_min, self.corruption_max))
        self.__corruption_min = value

    @property
    def corruption_max(self):
        """
        Property getter for corruption noise maximum value. Corruption is noise centered in
        the mean of corruption min and max.
        """
        return self.__corruption_max

    @corruption_max.setter
    def corruption_max(self, value):
        """
        Property setter for the corruption noise maximum value. Please be aware that this value
        must be greater than `corruption_min`, if set, or an error will be raised. It must be a float.
        """
        if value:
            self._checkFloat(value)
        if self.corruption_min:
            if value < self.corruption_min:
                raise ArgumentError("corruption_min ({}) is greater than corruption_max ({})".format(
                    self.corruption_min, self.corruption_max))
        self.__corruption_max = value

    @property
    def layers(self):
        """
        Property getter for layers. Define the number of the layer in this unit of the autoencoder.
        """
        return self.__layers

    @layers.setter
    def layers(self, value):
        """
        Property setter for the number of layers in this unit of the convolutional autoencoder.
        Must be an integer greater or equal to one.
        """
        if value:
            self._checkInt(value)
            if value < 1:
                raise ArgumentError("Number of layers must be positive")
        self.__layers = value

    @property
    def patch_size(self):
        """
        Property getter for the size of the patch. Weights tensor size is (convention "NHWC"):

        $$
        \mathrm{dim}(W) = \mathrm{input\_depth}\times\mathrm{patch\_size}^2\times\mathrm{input\_depth}
        $$
        """
        return self.__patch_size

    @patch_size.setter
    def patch_size(self, value):
        """
        Property setter for the convolutional weigths tensor patch (the two inner dimensions). Must
        be an `int` greater than one.
        """
        if value:
            self._checkInt(value)
            if value <= 1:
                raise ArgumentError("Values must be positive, greater than one")
        self.__patch_size = value

    @property
    def depth_increasing(self):
        """
        Property getter for the increasing depth. Final depth of the filter will be:

        $$
        \mathrm{output\_size} = \mathrm{input\_size} + \mathrm{depth\_increasing}
        $$
        """
        return self.__depth_increasing

    @depth_increasing.setter
    def depth_increasing(self, value):
        """
        Property setter for the depth incresing levels. Requires an `int` positive or 0.
        """
        if value:
            self._checkInt(value)
            if value < 0:
                raise ArgumentError("Values must be positive or zero")
        self.__depth_increasing = value

    @property
    def strides(self):
        """
        Property getter for the strides of the convolutional layer. The strides dimensions are define
        the reduction in size of the image:

        $$
        \mathrm{dim}(S) = 1 \times \mathrm{red}_{x} \times \mathrm{red}_{y} \times 1
        $$

        The first 1 works on batch elements number, while the last works on depth. With 1, no input
        is skipped.
        """
        return self.__strides

    @strides.setter
    def strides(self, value):
        """
        Property setter for the strides dimensions. It should be a list with four positive `int`,
        greater than 0
        """
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
        Property getter for the padding. It can be only "VALID" or "SAME".
        """
        return self.__padding

    @padding.setter
    def padding(self, value):
        """
        Property setter for padding: it can be only "SAME" or "VALID"
        """
        if value:
            self._checkStr(value)
            if value != "SAME" and value != "VALID":
                raise ArgumentError(
                    "Padding can be SAME or VALID. Received {}".format(value))
        self.__padding = value

    @property
    def prefix_name(self):
        """
        Prefix name for this particular layer. It will be usefull to identify it inside
        the TensorBoard visualization tool (will be used in front of all `name` property)
        """
        return self.__prefix_name

    @prefix_name.setter
    def prefix_name(self, value):
        """
        Propert setter of prefix name for this particular layer. It requires a `str` as input.
        """
        if value:
            self._checkStr(value)
            self.__prefix_name = value

    @property
    def residual_learning(self):
        """
        Property getter for residual learning flag. If this flag is `False`, the objective funtion in


        $$
        \mathrm{min}\,(y - x)^2 = \mathrm{min}\,(D(E(x)) - x)^2
        $$

        while, if `True`, it will try to learn:

        $$
        \mathrm{min}\,(D(E(x)) + x - x)^2 = \mathrm{min}\,(D(E(x)))^2
        $$

        (thus, the residuals only).
        """
        return self.__residual_learning

    @residual_learning.setter
    def residual_learning(self, value):
        """
        Define if learning the residuals: requires a `bool` ad input
        """
        if value:
            self._checkBool(value)
        self.__residual_learning = value

    # Other methods
    def _checkType(self, obj, tp):
        """
        Helper function, check with an `assert` the type specified.
        """
        assert type(obj) is tp, "%r is not of the correct type %r" % (obj, tp)

    def _checkStr(self, obj):
        """`str` check helper function"""
        self._checkType(obj, str)

    def _checkInt(self, obj):
        """`int` check helper function"""
        self._checkType(obj, int)

    def _checkFloat(self, obj):
        """`float` check helper function"""
        self._checkType(obj, float)

    def _checkBool(self, obj):
        """`bool` check helper function"""
        self._checkType(obj, bool)

    def _checkList(self, obj, size=0):
        """`list` check helper function. If a size is given check consistency"""
        self._checkType(obj, list)
        self._checkInt(size)
        if size > 0:
            assert len(obj) == size, "Size error for list: %d != %d" % (
                len(obj), size)

    def _checkHash(self, obj, keylist=None):
        """`dict` check helper function. If a keylist is given, check it contains all the keys"""
        self._checkType(obj, dict)
        if keylist is not None:
            self._checkList(keylist)
            for k in keylist:
                try:
                    obj[k]
                except KeyError:
                    raise KeyError(
                        "Key %r is not defined in object %r" % (k, obj))

    def _inspect(self):
        """Inspect"""
        print(self.__str__())

    def __str__(self):
        """Convert object into `str`"""
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
    """
    This class implements a simple convolutional autoencoder block. This block may contains different
    layers, and it is the the unit that is optimized. One single block is defined by an encoder mapping
    $E(\cdot)$ and a decoder mapping $D(\cdot)$.

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
    whilst, other blocks can be injected inside, thus leading to $h_{E} \neq{=} h_{D}$.

    If more $n$ layers are requested, there will be multiple layers that will share the same configuration
    in the form:
    $$
    E(\cdot) = (E_{n} \circ \dots \circ E_{1})(\cdot)
    $$
    and decoder in the form:
    $$
    D(\cdot) = (D_{n} \circ \dots \circ D_{1})(\cdot)
    $$
    For what concerns the single layers functions we have:
    $$
    E_{i}(\cdot) = \mathrm{LeakRelu}\left( \mathrm{Conv}(W_{i}, \cdot) + b_{E,i} \right)
    $$
    whilst the decoder is in the form:
    $$
    D_{i}(\cdot) = \mathrm{LeakRelu}\left( \mathrm{Deconv}(W^{T}_{i}, \cdot) + b_{D,i} \right)
    $$
    (the weights in the two operations are shared, in trasposed version). Interfaces of the block are:

     * `x`: input placeholder
     * `y`: output op
     * `h_enc`: innest layer op
     * `h_dec`: decoder input (it should be an op)
     * `weights`: a list of variables (weights for each encoding level)
     * `biases_encoder`: a list of biases variable for the encoder
     * `biases_decoder`: a list of biases for the decoder

    """

    def __init__(self, settings):
        """
        Initializes the block, using the helper class `ConvAutoEncSettings`, that contains all
        about the current block configuration. The use of **GPU** is defined on the basis of the
        application `FLAGS` variables.
        _Please consider that all high level attribute (withoust leading `__`) are inherit in this
        class as first level citizen, and accessible directly, but cannot be modified after object
        instanciation.
        """
        assert type(
            settings) is ConvAutoEncSettings, "Settings are not of type ConvAutoSettings"
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

        self.defineInput()
        self.defineEncoder()
        self.defineDecoder()
        self.defineCost()

    def __getattribute__(self, key):
        """Exposes all getter of the `ConvAutoEncSettings` in this class"""
        try:
            return super(ConvAutoEnc, self).__getattribute__('settings').__getattribute__(key)
        except AttributeError:
            return super(ConvAutoEnc, self).__getattribute__(key)

    def _corrupt(self, x, name="corruption"):
        """
        Corruption of a signal (helper function).

        The corruption is a multiplication between a ramdomly sampled tensor from an uniform
        distribution, with minimum value `corruption_min` and maximum value `corruption_max`
        The corruption acts as a simplified dropout layer.
        """
        with self.graph.as_default():
            with tf.name_scope("corruption"):
                return tf.mul(x, tf.cast(tf.random_unifor(shape=tf.shape(x),
                                                          minval=self.corruption_min,
                                                          maxval=self.corruption_max,
                                                          dtype=tf.int32)), name=name)

    def set_graph(self, graph):
        """
        Change the current `tf.Graph` (dafault: requested on object instantiation) with another one
        """
        assert type(graph) is tf.Graph
        self.graph = graph

    def leakRelu(self, x, alpha=0.2, name="leak-relu"):
        """
        The leaking relu is defined as follow:
        $$
        y = \dfrac{1}{2} \left( \alpha\,x + (1-\alpha)\,x \right)
        $$

        Options are $\alpha$ value and a name for the operation (`str`).
        """
        assert type(alpha) is float, "alpha must be a float"
        assert type(name) is str, "name must be a str"
        alpha_t = tf.constant(alpha, name=(name + "-constant"))
        with self.graph.as_default():
            with tf.name_scope(name):
                return 0.5 * ((1 + alpha_t) * x + (1 - alpha_t) * abs(x))

    def defineInput(self):
        """
        Creates the input placeholder and corruption operations if required
        """
        with self.graph.as_default():
            with tf.name_scope(self.prefix_name + "-input-layer"):
                x = tf.placeholder(tf.float32, self.input_shape, name=self.prefix_name + '-x')
                if self.corruption:
                    self.x = self._corrupt(x)
                else:
                    self.x = x
                self.add3summary(self.x, self.prefix_name + '-x')
        return self

    def defineEncoder(self):
        """
        Construct the encoder graph. Also assign `h_dec` as `h_enc`
        """
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
                    self.add2summary(W, name_W)
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
                self.h_dec = self.h_enc
                self.addXsummary(self.h_enc, self.prefix_name + '-h_enc')
                # self.add2summary(self.h, "h")

        return self

    def add1summary(self, var, name):
        """
        Add scalar elements to the summary, using name defined
        """
        assert type(name) is str, "Name must be a string"
        with tf.name_scope("summaries"):
            tf.scalar_summary(name, var)

    def add2summary(self, var, name):
        """
        Add histogram elements to the summary, using name defined
        """
        assert type(name) is str, "Name must be a string"
        with tf.name_scope("summaries"):
            pass
            tf.histogram_summary(name, var)

    def add3summary(self, var, name, batch=1):
        """
        Add image elements to the summary, using name defined
        """
        assert type(name) is str, "Name must be a string"
        assert type(batch) is int, "batch size must be an integer"
        assert batch > 0, "batch size must be positive"
        with tf.name_scope("summaries"):
            tf.image_summary(name, var, max_images=batch)

    def addXsummary(self, var, name, batch=1):
        """
        Add multi-dimensional elements to the summary, using name defined, and with `batch`
        samples from the batch dimension. The different dimension will be added as a monochromatic
        images, splitting layers by a numeric suffix.
        """
        assert type(name) is str, "Name must be a string"
        assert type(batch) is int, "batch size must be a integer"
        assert batch > 0, "batch size must be positive"
        for layer in range(0, var.get_shape()[3]):
            name_l = "%s-L%d" % (name, layer)
            tf.image_summary(name_l, var[:, :, :, layer:layer + 1], max_images=batch)

    def defineDecoder(self):
        """
        Defines the decoder layer. At the end defines also `y` to reflect residual learning
        or not.
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
                    h_layer = self.leakRelu(tf.add(
                        tf.nn.conv2d_transpose(
                            x_current, W, shape, strides=self.strides, padding=self.padding,
                            name=name_deconv),
                        B, name=name_sum
                    ))
                    x_current = self.leakRelu(h_layer, name=name_out)
                if self.residual_learning:
                    self.y = x_current + self.x
                else:
                    self.y = x_current
                self.add3summary(self.y, self.prefix_name + '-y')
        return self

    def defineCost(self):
        """
        Define the cost function that must be otpimized for this block, as minimization
        of the sum on the square error between input and output.
        """
        if (self.x is None) or (self.h is None) or (self.y is None):
            raise RuntimeError("You cannot define a cost, if you not define output")
        with self.graph.as_default():
            with tf.name_scope(self.prefix_name + '-cost-function'):
                self.error = tf.reduce_sum(
                    tf.square(self.y - self.x), name=self.prefix_name + '-error-definition')
                self.add1summary(self.error, self.prefix_name + '-error')
        return self


class CombinedAutoencoder:

    def __init__(self, config_tuple):
        assert type(config_tuple) is tuple, "Required a tuple of ConvAutoEncSettings"
        self.caes = []
        for c in config_tuple:
            assert type(c) is ConvAutoEncSettings, "Element in config tuple is not ConvAutoEncSettings"
            self.caes.append(ConvAutoEnc(c))

    def len(self):
        return len(self.caes)
