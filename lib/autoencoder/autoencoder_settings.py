#!/usr/bin/env python3

r"""
.. module:: autoencoder_blocks
   :platform: Linux
   :synopsis: Implementation of Convolutional Autoencoder Settings

Contains the implementation of the single of the convolutional autoencoder settings
interface.

.. :moduleauthor: Matteo Ragni
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autoencoder_helpers import ArgumentError
import tensorflow as tf

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
