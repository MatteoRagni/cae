#!/usr/bin/env python3

import cmd
import tensorflow as tf
import autoencoder
import numpy as np
import os
import timer
from data_handler import DataHandler
from six.moves import cPickle as pickle
from argparse import ArgumentParser

#   ___                              _   _    _            ___     _            __
#  / __|___ _ __  _ __  __ _ _ _  __| | | |  (_)_ _  ___  |_ _|_ _| |_ ___ _ _ / _|__ _ __ ___
# | (__/ _ \ '  \| '  \/ _` | ' \/ _` | | |__| | ' \/ -_)  | || ' \  _/ -_) '_|  _/ _` / _/ -_)
#  \___\___/_|_|_|_|_|_\__,_|_||_\__,_| |____|_|_||_\___| |___|_||_\__\___|_| |_| \__,_\__\___|
class ConvAutoEncArgParse(ArgumentParser):

    def __init__(self):
        super(ConvAutoEncArgParse, self).__init__(prog="caes-bin",
            description="Stacked Convolutional Autoencoder",
            epilog="Matteo Ragni, David Windridge, Paolo Bosetti - 2016")

        self.identifier = timer.timestring()

        # Id options
        self.add_argument('-id', '--identifier', dest='identifier', type=str, nargs=1, default=[self.identifier],
            help='Training identifier - time base [{}]'.format(self.identifier))

        # File options
        self.add_argument('-w', '--workspace', dest='workspace', type=str, nargs=1, required=True,
            help='Workspace directory. Will contain run files generated during training and inference')
        self.add_argument('-d', '--dataset', dest='dataset', type=str, nargs=1, required=True,
            help='Dataset directory. Must contain training and inference datasets')
        self.add_argument('-m', '--model', dest='model', type=str, nargs=1, required=True,
            help='Model binary file')

        self.add_argument('--training-id', dest='training_id', type=str, nargs=1,
            default=['train-' + self.identifier],
            help='Run directory will be created inside workspace')
        self.add_argument('-s', '--save', dest='save_file', type=str, nargs=1,
            help='Checkpoint saving file')
        self.add_argument('-l', '--load', dest='load_file', type=str, nargs=1,
            help='Checkpoint loading file (will skip training)')

        # Training options
        self.add_argument('-bs', '--batch-size', dest='batch_size', type=int, nargs=1, default=10,
            help='Batch size (number of examples for learning step)')
        self.add_argument('-sz', '--steps', dest='steps', type=int, nargs=1, default=10,
            help='Number of reiteratios on a single batch')
        self.add_argument('-bb', '--batch-block', dest='batch_block', type=int, nargs=1, default=8,
            help='Number of blocks of batches to be loaded (1 batch = 1000 figures)')

        self.add_argument('--learning-rate', dest='learn_rate', type=float, nargs=1, default=0.001,
            help='Learning rate hyper-parameters')
        self.add_argument('--residual-learning', dest='residuals', action='store_true',
            help='Enable residual learning (NO -> y = f(g(x)), YES -> y = f(g(x)) + x) [NO]')

        # Other Options
        # self.add_argument('--notify', dest='notification', action='store_true', default=False,
        #     help='Enable notifications using system telegram bot, at the end of the learning')
        return None

    def run(self):
        self.initialize().parse_args()
        return self._handleFiles()._handleTraining()

    def _handleFiles(self):
        self.training_file  = os.path.join(' '.join(self.dataset), 'training.pickle')
        self.inference_file = os.path.join(' '.join(self.dataset), 'inference.pickle')
        self.model = os.path.join(' '.join(self.model))
        self.load_file = os.path.join(' '.join(self.load_file))
        self.save_file = os.path.join(' '.join(self.save_file))

        assert os.path.isfile(self.training_file), "Dataset file {} does not exist".format(self.training_file)
        assert os.path.isfile(self.inference_file), "Inference file {} does not exist".format(self.inference_file)
        assert os.path.isfile(self.model), "Model file {} does not exist".format(self.model)
        assert os.path.isfile(self.load_file), "Restored model {} does not exist".format(self.load_file)

        self.workspace = os.path.join(' '.join(self.workspace))
        self.training_dir = os.path.join(self.workspace, self.training_dir)
        return self

    def _handleTraining(self):
        if type(self.batch_size) is list:
            self.batch_size = self.batch_size[0]
        assert self.batch_size > 0, "Batch size must be positive"
        if type(self.steps) is list:
            self.steps = self.steps[0]
        assert self.step_size > 0, "Batch size must be positive"
        if type(self.batch_block) is list:
            self.batch_block = self.batch_block[0]
        assert self.batch_block > 0, "Batch size must be positive"
        if type(self.learn_rate) is list:
            self.learn_rate = self.learn_rate[0]
        assert self.learn_rate > 0, "Batch size must be positive"
        return self

#  _                      _             ___ _        _ _
# | |   ___ __ _ _ _ _ _ (_)_ _  __ _  / __| |_  ___| | |
# | |__/ -_) _` | '_| ' \| | ' \/ _` | \__ \ ' \/ -_) | |
# |____\___\__,_|_| |_||_|_|_||_\__, | |___/_||_\___|_|_|
#                               |___/
class ConvAutoEncShell(cmd.Cmd):
    r"""
    A simple shell to check the situation of the AutoEncoder after the training
    """

    intro = "AutoEncoder Stack post-learn shell\n  Matteo Ragni, David Windridge - 2016\n  Type help or ? to ls"
    prompt = ">>> "

    def __init__(self, config):
        assert type(config) is ConvAutoEncArgParse, "Configuration must be a ConvAutoEncArgParse"

        super(ConvAutoEncShell, self).__init__()
        config.parse_args()
        self.flags = config
        self.hallucinate = {"on": 1.0, "off": 0.0}
        self.config = {"objects": 3, "positions": 25}

        self.loadDataset()

        self.inner_shape = None
        self._loadModel()
        self.writer = self._createWriter(self.flags.training_dir)
        if not self._existRestore():
            self._learnModel()

        self.empty_x = np.full(
            tuple(self.model.caes[0].x.get_shape().as_list()), 0, dtype=np.float32)
        self.cmdloop()

    def print_info(self, s):
        print(s)

    def print_warn(self, s):
        print('\033[93m' + s + '\033[0m')

    def print_err(self, s):
        print('\033[91m' + s + '\033[0m')

    def print_done(self, s):
        print('\033[92m' + s + '\033[0m')

    def print_learn(self, cb, cnt, res, n):
        print("RUNNING BATCD: %d (c: %d, e: %.5e) on layer %d" % (cb, cnt, res, n))

    def _loadModel(self):
        self.print_info("Loading the graph")
        try:
            with open(self.flags.model, "rb") as fp:
                sets = pickle.load(fp)
                for s in sets:
                    s.input_shape[0] = self.flags.batch_size
                self.model = autoencoder.ConvAutoEncStack(sets, self.flags.learning_rate)
                self.graph = self.model.graph
                self.session = self.model.session
                self.merged = tf.merge_all_summaries()
                self._createWriter(self.flags.training_dir)
                self.print_done("Model loaded")
                return self
        except Exception as e:
            self.print_err(("Cannot load model: {}").format(e))
            exit(1)

    def _existRestore(self):
        try:
            if self.flags.load_file is "":
                return False
            else:
                self.print_info("Loading previous session: {}".format(self.flags.load_file))
                self._reloadLearning()
                return True
        except Exception as e:
            self.print_err("Error: {}".format(e))
            return None

    def _realoadLearning(self):
        try:
            self.model.restore(self.flags.load_file)
            return self
        except Exception as e:
            self.print_err("Error: {}".format(e))
            exit(1)

    def _createWriter(self, fl):
        try:
            return tf.train.SummaryWriter(fl, self.graph)
        except Exception as e:
            self.print_err("Error: {}".format(e))
            return None

    def _loadDataset(self):
        try:
            self.dataset = DataHandler(self.flags.dataset_file,
                self.flags.inference_file,
                tuple(self.model.caes[0].input_shape))
            return self.dataset
        except Exception as e:
            self.print_err("Error: {}".format(e))
            return None


    def _learnModel(self):
        try:
            counter = 0
            for session, n, cae, x in self.model.trainBlocks():
                result = [None, 10e10]
                with tf.name_scope("TRAINING-%d" % n):
                    current_batch = -1
                    for batch_no, dataset in self.dataset(self.flags.batch_size, self.flags.batch_block):
                        # Printing
                        if current_batch != batch_no:
                            current_batch = batch_no
                            self.print_learn(current_batch, counter, result[1], n)

                        # Actual learining steps
                        with timer.Timer():
                            for step in range(0, self.flags.steps):
                                result = self.session.run([cae.optimizer], feed_dict={x: dataset[0]})
                                counter += 1

                        # Loss print
                        losses = 0
                        losses_string = "\033[94m%d\033[0m" % counter
                        for im in dataset:
                            loss = self.session.run(self.model.error, feed_dict={x: im})
                            losses_string += "\t%.5e" % loss
                            losses += loss
                        losses_string += "\t\033[94m%.5e\033[0m" % losses
                        print(losses_string)

                        result = self.session.run([self.merged, cae.error], feed_dict={x: dataset[0]})
                        self.writer.add_summary(result[0], counter)
            self.print_done("Training complete")
        except Exception as e:
            self.print_err("Error: {}".format(e))

        return self



    # COMMAND IMPLEMENTATION
    def do_infer_single(self, arg):
        r"""
        Loads support image with one object. It must be provided in the form:
        LOAD_SINGLE x y with x [1..3] and y [1..25]
        """
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
        Configuration is a pickle file
        """
        try:
            if not os.path.isfile(arg):
                raise Exception("file does not exist")
            with open(arg, "rb") as f:
                conf = pickle.load(f)
                return conf # TODO: continuare

        except Exception as e:
            print("ERROR: {}".format(e))

    def do_infer_double(self, arg):
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

            self.writer = tf.train.SummaryWriter(str(arg), self.config["stack"].graph)
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
