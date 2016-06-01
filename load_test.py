# Importing libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from glob import glob
import random
from datetime import datetime
import time

from six.moves import cPickle as pickle

import tensorflow as tf

from imp import reload
import autoencoder


# Some definitions on the engine
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir',
                           '/tmp/training_nn',
                           "Training directory")

tf.app.flags.DEFINE_integer('max_steps',
                            1000000,
                            "Number of iterations")

tf.app.flags.DEFINE_boolean('log_defice_placement',
                            False,
                            "Logging of the use of my device")


reload(autoencoder)

sets = autoencoder.ConvAutoEncSettings()

# Define configurations for the Convolutional Autoencoder
# To get a preview of the options that can be set, print(sets)
sets.input_shape = [1000, 277, 277, 3]
sets.corruption = False
sets.layers = 1
sets.patch_size = 29
sets.depth_increasing = 2
sets.strides = [1, 2, 2, 1]
sets.padding = "SAME"
sets.cuda_enabled = True

print(sets)

# Loading the Convolutional Autoencoder
cae = autoencoder.ConvAutoEnc(sets)
