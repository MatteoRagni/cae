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
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autoencoder_helpers import ArgumentError
from autoencoder_settings import ConvAutoEncSettings
from autoencoder_blocks import ConvAutoEnc
from autoencoder_stack import ConvAutoEncStack

from autoencoder import ArgumentError
from autoencoder import ConvAutoEncSettings
from autoencoder import ConvAutoEnc
from autoencoder import ConvAutoEncStack

__all__ = ['autoencoder_helpers', 'autoencoder_settings', 'autoencoder_blocks', 'autoencoder_stack']
