#!/usr/bin/env python3

# Importing libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# import os
# from glob import glob
# import random


class ArgumentError(ValueError):
    pass

#  ___      _   _   _                 ___ _
# / __| ___| |_| |_(_)_ _  __ _ ___  / __| |__ _ ______
# \__ \/ -_)  _|  _| | ' \/ _` (_-< | (__| / _` (_-<_-<
# |___/\___|\__|\__|_|_||_\__, /__/  \___|_\__,_/__/__/
#                         |___/


class ConvAutoEncSettings(object):

    def __init__(self):
        self.__input_shape      = None
        self.__corruption       = None
        self.__corruption_min   = None  # 0
        self.__corruption_max   = None  # 0
        self.__depth_increasing = None
        self.__layers           = None
        self.__patch_size       = None
        self.__strides          = None
        self.__padding          = None  # SAME / VALID
        self.__cuda_enabled     = None

    # Check methods
    def checkComplete(self):
        if not(self.input_shape      is not None and
               self.corruption       is not None and
               self.layers           is not None and
               self.patch_size       is not None and
               self.strides          is not None and
               self.padding          is not None and
               self.cuda_enabled     is not None and
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
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, value):
        if value:
            self._checkList(value)
            for v in value:
                self._checkInt(v)
                if v <= 0:
                    raise ArgumentError("Values must be positive")
        self.__input_shape = value

    @property
    def corruption(self):
        return self.__corruption

    @corruption.setter
    def corruption(self, value):
        if value:
            self._checkBool(value)
        if value is False:
            self.corruption_min = self.corruption_max = 0
        self.__corruption = value

    @property
    def corruption_min(self):
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

    @property
    def corruption_max(self):
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

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, value):
        if value:
            self._checkInt(value)
            if value < 1:
                raise ArgumentError("Number of layers must be positive")
        self.__layers = value

    @property
    def patch_size(self):
        return self.__patch_size

    @patch_size.setter
    def patch_size(self, value):
        if value:
            self._checkInt(value)
            if value <= 0:
                raise ArgumentError("Values must be positive")
        self.__patch_size = value

    @property
    def depth_increasing(self):
        return self.__depth_increasing

    @depth_increasing.setter
    def depth_increasing(self, value):
        if value:
            self._checkInt(value)
            if value <= 0:
                raise ArgumentError("Values must be positive")
        self.__depth_increasing = value

    @property
    def strides(self):
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
        return self.__padding

    @padding.setter
    def padding(self, value):
        if value:
            self._checkStr(value)
            if value != "SAME" and value != "VALID":
                raise ArgumentError(
                    "Padding can be SAME or VALID. Received {}".format(value))
        self.__padding = value

    @property
    def cuda_enabled(self):
        return self.__cuda_enabled

    @cuda_enabled.setter
    def cuda_enabled(self, value):
        if value:
            self._checkBool(value)
        self.__cuda_enabled = value

    # Other methods
    def _checkType(self, obj, tp):
        assert type(obj) is tp, "%r is not of the correct type %r" % (obj, tp)

    def _checkStr(self, obj):
        self._checkType(obj, str)

    def _checkInt(self, obj):
        self._checkType(obj, int)

    def _checkFloat(self, obj):
        self._checkType(obj, float)

    def _checkBool(self, obj):
        self._checkType(obj, bool)

    def _checkList(self, obj, size=0):
        self._checkType(obj, list)
        self._checkInt(size)
        if size > 0:
            assert len(obj) == size, "Size error for list: %d != %d" % (
                len(obj), size)

    def _checkHash(self, obj, keylist=None):
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
        print(self.__str__())

    def __str__(self):
        return "  Convolutional Autoencoder Settings"                   + "\n" + \
               "--------------------------------------"                 + "\n" + \
               " - input_shape      = {}".format(self.input_shape)      + "\n" + \
               " - corruption       = {}".format(self.corruption)       + "\n" + \
               " - corruption_min   = {}".format(self.corruption_min)   + "\n" + \
               " - corruption_max   = {}".format(self.corruption_max)   + "\n" + \
               " - layers           = {}".format(self.layers)           + "\n" + \
               " - depth_increasing = {}".format(self.depth_increasing) + "\n" + \
               " - patch_size       = {}".format(self.patch_size)       + "\n" + \
               " - strides          = {}".format(self.strides)          + "\n" + \
               " - padding          = {}".format(self.padding)          + "\n" + \
               " - cuda_enabled     = {}".format(self.cuda_enabled)


#   ___                 _      _   _              _       _                            _
#  / __|___ _ ___ _____| |_  _| |_(_)___ _ _     /_\ _  _| |_ ___  ___ _ _  __ ___  __| |___ _ _
# | (__/ _ \ ' \ V / _ \ | || |  _| / _ \ ' \   / _ \ || |  _/ _ \/ -_) ' \/ _/ _ \/ _` / -_) '_|
#  \___\___/_||_\_/\___/_|\_,_|\__|_\___/_||_| /_/ \_\_,_|\__\___/\___|_||_\__\___/\__,_\___|_|
class ConvAutoEnc(object):

    def __init__(self, settings):
        assert type(
            settings) is ConvAutoEncSettings, "Settings are not of type ConvAutoSettings"
        self.settings = settings
        self.settings.checkComplete()

        self.graph   = tf.Graph()
        self.x       = None
        self.h       = None
        self.y       = None
        self.weights = None
        self.latents = None
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
        # self.createSession()

    def __getattribute__(self, key):
        try:
            return super(ConvAutoEnc, self).__getattribute__('settings').__getattribute__(key)
        except AttributeError:
            return super(ConvAutoEnc, self).__getattribute__(key)

    def _corrupt(self, x, name="corruption"):
        with self.graph.as_default():
            with tf.name_scope("corruption"):
                return tf.mul(x, tf.cast(tf.random_unifor(shape=tf.shape(x),
                                                          minval=self.corruption_min,
                                                          maxval=self.corruption_max,
                                                          dtype=tf.int32)), name=name)

    def leakRelu(self, x, alpha=0.2, name="leak-relu"):
        alpha_t = tf.constant(alpha, name=(name + "-constant"))
        with self.graph.as_default():
            with tf.name_scope(name):
                return 0.5 * ((1 + alpha_t) * x + (1 - alpha_t) * abs(x))

    def defineInput(self):
        with self.graph.as_default():
            with tf.name_scope("input-layer"):
                x = tf.placeholder(tf.float32, self.input_shape, name='x')
                if self.corruption:
                    self.x = self._corrupt(x)
                else:
                    self.x = x
                self.add3summary(self.x, "x")
        return self

    def defineEncoder(self):
        if self.x is None:
            raise RuntimeError("You must define input to define the encoder")
        with self.graph.as_default():
            with tf.name_scope("encoder"):
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
                    name_W    = "weight-%d"              % layer
                    name_B    = "enc-bias-%d"            % layer
                    name_conv = "encoder-convolution-%d" % layer
                    name_sum  = "encoder-sum-%d"         % layer
                    name_out  = "encoder-out-%d"         % layer

                    W = tf.Variable(tf.truncated_normal(
                        patch, stddev=0.1), name=name_W)
                    self.add2summary(W, name_W)
                    B = tf.Variable(tf.zeros([new_depth]), name=name_B)
                    self.add2summary(B, name_B)

                    self.weights.append(W)
                    self.biases_encoder.append(B)

                    h_layer = tf.add(
                        tf.nn.conv2d(x_current, W, strides=self.strides,
                                     padding=self.padding, name=name_conv),
                        B, name=name_sum
                    )
                    x_current = self.leakRelu(h_layer, name=name_out)
                self.h = x_current
                self.addXsummary(self.h, "h")
                # self.add2summary(self.h, "h")

        return self

    def add1summary(self, var, name):
        assert type(name) is str, "Name must be a string"
        with tf.name_scope("summaries"):
            tf.scalar_summary(name, var)

    def add2summary(self, var, name):
        assert type(name) is str, "Name must be a string"
        with tf.name_scope("summaries"):
            pass
            tf.histogram_summary(name, var)

    def add3summary(self, var, name, batch=1):
        assert type(name) is str, "Name must be a string"
        assert type(batch) is int, "batch size must be an integer"
        assert batch > 0, "batch size must be positive"
        with tf.name_scope("summaries"):
            tf.image_summary(name, var, max_images=batch)

    def addXsummary(self, var, name, batch=1):
        assert type(name) is str, "Name must be a string"
        assert type(batch) is int, "batch size must be a integer"
        assert batch > 0, "batch size must be positive"
        for layer in range(0, var.get_shape()[3]):
            name_l = "%s-L%d" % (name, layer)
            tf.image_summary(name_l, var[:, :, :, layer:layer + 1], max_images=batch)

    def defineDecoder(self):
        if (self.x is None) or (self.x is None):
            raise RuntimeError("To define a decoder you must define the encoder")
        self.biases_decoder = []
        with self.graph.as_default():
            with tf.name_scope("decoder"):
                x_current = self.h
                for current_layer in range(0, self.layers):
                    layer       = self.layers - (current_layer + 1)
                    name_deconv = "decoder-deconvolution-%i" % layer
                    name_sum    = "decoder-sum-%i"           % layer
                    name_out    = "decoder-out-%i"           % layer
                    name_B      = "dec-bias-%d"              % layer

                    W = self.weights[layer]
                    B = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]), name=name_B)
                    self.add2summary(B, name_B)

                    self.biases_decoder.append(B)

                    shape = self.shapes[layer]
                    h_layer = self.leakRelu(tf.add(
                        tf.nn.conv2d_transpose(
                            x_current, W, shape, strides=self.strides, padding=self.padding, name=name_deconv),
                        B, name=name_sum
                    ))
                    x_current = self.leakRelu(h_layer, name=name_out)
                self.y = x_current + self.x # RESIDUAL LEARNING MODIFICATION
                self.add3summary(self.y, "y")
        return self

    def defineCost(self):
        if (self.x is None) or (self.h is None) or (self.y is None):
            raise RuntimeError("You cannot define a cost, if you not define output")
        with self.graph.as_default():
            with tf.name_scope("cost-function"):
                self.error = tf.reduce_sum(
                    tf.square(self.y - self.x), name="error-definition")
                self.add1summary(self.error, "error")
        return self

    def createSession(self, learning_rate=0.01):
        assert type(learning_rate) is float, "Learning Rate must be a float"
        assert learning_rate > 0, "Learning rate must be positive"

        with self.graph.as_default():
            with tf.name_scope("define-optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.error)
        self.session = tf.Session(graph=self.graph)
        with self.session.as_default():
            self.session.run(tf.initialize_all_variables())
        return self

    def trainBatch(self, training_tensor):
        assert self.session is None, "Session is not initialized"
        assert type(training_tensor) is np.ndarray, "Training tensor is not Numpy Tensor"
        with self.session.as_default():
            self.session.run(self.optimizer, feed_dict={self.x: training_tensor})
            return {'cost': self.session.run(self.error, feed_dict={self.x: training_tensor}),
                    'y': self.session.run(self.y, feed_dict={self.x: training_tensor}),
                    'h': self.session.run(self.h, feed_dict={self.x: training_tensor})}
