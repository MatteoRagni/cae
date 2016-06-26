#!/usr/bin/env python3

import cmd
# import sys
import tensorflow as tf


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

    def do_load_duble(self, arg):
        r"""
        Loads support image with double object. It must be provided in the form:
        LOAD_SINGLE x y z w with (x, y) in [1..%d] and (z, w) in  [1..%d]
        """ % (self.config["objects"], self.config["positions"])
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
            pos = "/tmp/training_nn/"
            self.writer = tf.train.SummaryWriter(pos + str(arg), self.config["stack"].graph)
        except Exception as e:
            print("Cannot create writer: {}".format(e))

    def do_exit(self, arg):
        print("Bye!")
        return True

    # LOW LEVEL ROUTINES, NOT CALLABLE
    def select(self, call):
        r"""
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
            print("Something went really wrong: {}".format(e))
            return

    def parse(self, arg):
        return tuple(map(int, arg.split()))
