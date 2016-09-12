#!/usr/bin/env python3

import cmd
# import sys
import tensorflow as tf
import numpy as np

# Load configuration
import os
import json

"""
class NewConvAutoEncShell(cmd.Cmd):
    r"" " <-- TODO
    New version of the ConvAutoEnc shell. This shell is spawned
    before the training and it is actually responsible for the
    training itself. It can receive some command from command line
    input in a form of `dict` with some key defined:

     - json_file: contains the json with the architecture description
     - datasets: an array that contains dataset and support file
     - batchsize
    "" " <-- TODO

    from data_handler import DataHandler
    from timer import Timer


    def __init__(self, config=None):
        if config is not None:
            assert type(config) is dict, "config must be a dict"
            try:

            except KeyError:
                pass
"""


class ConvAutoEncShell(cmd.Cmd):
    """
    A simple shell to check the situation of the AutoEncoder after the training
    """

    intro = """
AutoEncoder Stack post-learn shell
  Matteo Ragni, David Windridge - 2016
  Type help or ? to ls
"""
    prompt = ">>> "
    file = None

    def __init__(self, config):
        assert type(config) is dict, "Configuration must be a dictionary"
        # assert hasattr(callback, '__call__'), "Callback function does not work"

        super(ConvAutoEncShell, self).__init__()
        self.config = config
        self.writer = None
        self.hallucinate = {"on": 1.0, "off": 0.0}
        self.inner_shape = None
        self.empty_x = np.full(
            tuple(self.config["stack"].caes[0].x.get_shape().as_list()), 0, dtype=np.float32)

    # COMMAND IMPLEMENTATION
    def do_load_single(self, arg):
        r"""
        Loads support image with one object. It must be provided in the form:
        LOAD_SINGLE x y with x [1..%d] and y [1..%d]
        """ % (self.config["objects"], self.config["positions"])
        try:
            arg = self.parse(arg)
            assert type(arg) is tuple, "wrong argument passed: {}"
            assert type(arg[0]) is int, "first argument is not an int"
            assert type(arg[1]) is int, "second argument is not an int"

            if not (arg[0] - 1) in range(0, self.config["objects"]):
                print("Element not in range")
                return True

            if not (arg[1] - 1) in range(0, self.config["positions"]):
                print("Position not in range")
                return True

            call = ["99", "99", "99"]
            call[arg[0] - 1] = "%02d" % (arg[1] - 1)
            call = tuple(call)

            self.select(call)
        except Exception as e:
            print("ERROR: {}".format(e))

    def do_configure(self, arg):
        r"""
        Load configuration to define a new Convolutional autoencoder.
        Configuration is a json file
        """
        try:
            if not os.path.isfile(arg):
                raise Exception("file does not exist")
            with open(arg, "r") as f:
                conf = json.load(f)
                return conf # TODO: continuare

        except Exception as e:
            print("ERROR: {}".format(e))

    def do_load_duble(self, arg):
        r"""
        Loads support image with double object. It must be provided in the form:
        LOAD_SINGLE x y z w with (x, y) in [1..num_el] and (z, w) in  [1..num_pos]
        """
        try:
            arg = self.parse(arg)
            assert type(arg) is tuple, "wrong argument passed"
            for a in arg:
                assert type(a) is int, "first argument is not an int"

            if not (arg[0] - 1) in range(0, self.config["objects"]):
                print("Element not in range")
                return True
            if not (arg[1] - 1) in range(0, self.config["objects"]):
                print("Element not in range")
                return True
            if not (arg[2] - 1) in range(0, self.config["positions"]):
                print("Position not in range")
                return True
            if not (arg[3] - 1) in range(0, self.config["positions"]):
                print("Position not in range")
                return True

            call = ["99", "99", "99"]
            call[arg[0] - 1] = "%02d" % (arg[2] - 1)
            call[arg[1] - 1] = "%02d" % (arg[3] - 1)
            call = tuple(call)

            self.select(call)
        except Exception as e:
            print("ERROR: {}".format(e))

    def do_new_writer(self, arg):
        """
        Creates a new writer on which save some data
        """
        try:
            if self.writer is not None:
                print("I will close the previous writer first")
                self.writer.close()

            pos = "/tmp/training_nn/"
            self.writer = tf.train.SummaryWriter(pos + str(arg), self.config["stack"].graph)
            print("New writer created")
        except Exception as e:
            print("Cannot create writer: {}".format(e))

    def do_hallucination(self, layers):
        """
        Rewrites one of the inner dimensions of the autoencoder to check hallucinations.
        The off value and on value are changed using.

        :warning: DOES NOT WORK.
        """
        try:
            if self.writer is None:
                print("Writer not defined. Please create a new writer first")
                return

            # p = self.config["stack"].len() - 1
            h = self.config["stack"].inception
            x = self.config["stack"].caes[0].x
            if self.inner_shape is None:
                self.inner_shape = tuple(h.get_shape().as_list())
            layers = self.parse(layers)
            for k in layers:
                if not k < self.inner_shape[3]:
                    print("k ({}) too big (> {})".format(k, self.inner_shape[3]))
            new_h = np.full(self.inner_shape, self.hallucinate["off"], dtype=np.float32)
            act_h = np.full((self.inner_shape[0], self.inner_shape[1], self.inner_shape[2], 1), self.hallucinate["on"], dtype=np.float32)
            for k in layers:
                new_h[:, :, :, k:k + 1] = act_h

            self.config["counter"] += 1
            print("Hallucinating...")
            result = self.config["session"].run([self.config["summary"]], feed_dict={h: new_h, x: self.empty_x})
            print("Wrinting on tensorboard...")
            self.writer.add_summary(result[0], self.config["counter"])
            self.writer.flush()
            print("Completed - Reload TensorBoard")

        except Exception as e:
            print("Error: {}".format(e))

    def do_hallucination_set(self, arg):
        """
        Defines hallucinate on and off values
        """
        try:
            arg = arg.split()
            if arg[0] is "on":
                self.hallucinate["on"] = float(arg[1])
            if arg[0] is "off":
                self.hallucinate["off"] = float(arg[1])
        except Exception as e:
            print("Error: {}".format(e))

    def do_exit(self, arg):
        print("Bye!")
        return True

    # LOW LEVEL ROUTINES, NOT CALLABLE
    def select(self, call):
        """
        Perform the actual selection of the support image from the support database
        """
        try:
            if self.writer is None:
                print("Writer not defined. Please create a new writer first")
                return

            test = self.config["dh"].test_image(call)
            self.config["counter"] += 1
            print("Evaluating...")
            result = self.config["session"].run([self.config["summary"]], feed_dict={self.config["stack"].caes[0].x: test})
            print("Wrinting on tensorboard...")
            self.writer.add_summary(result[0], self.config["counter"])
            self.writer.flush()
            print("Completed - Reload TensorBoard")
        except Exception as e:
            print("Error: {}".format(e))
            return

    def parse(self, arg):
        return tuple(map(int, arg.split()))
