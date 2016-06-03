# Importing libraries
# For using my GPU I have to do that...
# export TF_MIN_GPU_MULTIPROCESSOR_COUNT=2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy as np
# import os
# from glob import glob
# import random
# from datetime import datetime
# import time

from six.moves import cPickle as pickle

import tensorflow as tf

# from imp import reload
import autoencoder
from timer import Timer


# Some definitions on the engine
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir',
                           '/tmp/training_nn',
                           "Training directory")

tf.app.flags.DEFINE_integer('max_steps',
                            10,
                            "Number of iterations")

# tf.app.flags.DEFINE_boolean('log_device_placement',
#                             True,
#                             "Logging of the use of my device")

tf.app.flags.DEFINE_float('learning_rate',
                          0.01,
                          "Optimizer learning rate")

# reload(autoencoder)

sets = autoencoder.ConvAutoEncSettings()

# Define configurations for the Convolutional Autoencoder
# To get a preview of the options that can be set, print(sets)
sets.input_shape = [50, 227, 227, 3]
sets.corruption = False
sets.layers = 1
sets.patch_size = 29
sets.depth_increasing = 4
sets.strides = [1, 2, 2, 1]
sets.padding = "SAME"
sets.cuda_enabled = False

print(sets)
print("Running on TensorFlow [%s]" % tf.__version__)

# Loading the Convolutional Autoencoder
cae = autoencoder.ConvAutoEnc(sets)
with open("./dataset/data.pickle", "rb") as f:
    no_batch = pickle.load(f)
    counter = 0

    # cae.trainBatch(first_run)
    with tf.Session(graph=cae.graph) as session:
        try:
            cae.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, name="Optimizer").minimize(cae.error)
            session.run(tf.initialize_all_variables())
            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter(FLAGS.train_dir, session.graph)

            offset   = 50
            length   = 1000

            # Waits before starting the
            try:
                input("Press a key to continue...")
            except SyntaxError:
                pass
            with tf.name_scope("Training"):
                for reloaded in range(0, no_batch - 4):
                    first_run = pickle.load(f)
                    for i in range(0, length // offset):
                        with Timer():
                            init = i * offset
                            ends = (i + 1) * offset
                            for tt in range(0, FLAGS.max_steps):
                                result = session.run([cae.optimizer], feed_dict={cae.x: first_run[init:ends, :, :, :]})
                                counter += 1
                            result = session.run([merged, cae.error], feed_dict={cae.x: first_run[init:ends, :, :, :]})
                            writer.add_summary(result[0], counter)
                            print("Error at step %i is: %5.10f" % (counter, result[1]))
        finally:
            if writer is not None:
                writer.close()
            if session is not None:
                session.close()

print("StopHere")
