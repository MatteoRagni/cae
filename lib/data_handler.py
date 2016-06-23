#!/usr/bin/env python3

r"""
.. module:: data_handler
   :platform: Linux
   :synopsis: Helper class for handling datasets

This is a support module for the hanndling of our complex dataset. Currently the dataset is
composed of two elements:

 * a file that contains a multiple written data, the first th config, all the others the batches
 * a file that contains the support images

.. :moduleauthor: Matteo Ragni
"""

from __future__ import print_function

import numpy as np
from glob import glob
import random
import re
from os.path import isfile

from scipy import ndimage
from six.moves import cPickle as pickle


#  ___       _        ___                             _
# |   \ __ _| |_ __ _| _ \_ _ ___ _ __  __ _ _ _ __ _| |_ ___ _ _
# | |) / _` |  _/ _` |  _/ '_/ -_) '_ \/ _` | '_/ _` |  _/ _ \ '_|
# |___/\__,_|\__\__,_|_| |_| \___| .__/\__,_|_| \__,_|\__\___/_|
#                                |_|
class DataPreparator:
    r"""
    This is an utility class, used from command line for handling the creation of the pickles
    that contains the images.

    :warning: as for now this class is locked with three elements due to short developing time.
    """
    def __init__(self):
        r"""
        Configuration for the final pickle file. Those configuration must be changed manually.
        There are several voices:

         * :py:attr:`header`: it will be the first thing that will be written on the output pickle
           file.
         * the batches are the following elements in
        """
        self.header = {
            "total": 0,    # auto
            "batches": 0,  # auto
            "batch_size": 1000,
            "shape": (161, 161, 1),
            "objects": 3,
            "desc": {
                "desc": "Dataset, depth 1, all permutations = [123, 1, 2, 3, 12, 13, 23]",
                "total": "Total number of scenes in the dataset",
                "batches": "Number of batches in the dataset",
                "batch_size": "Number of elements in a single batch",
                "shape": "Shape of a single element",
                "objects": "Number of object in a image"
            }
        }
        self.single_dir = "dataset_1"
        self.double_dir = "dataset_2"
        self.triple_dir = "dataset_3"

        self.out_file = "dataset_complete.pickle"
        self.support_file = "dataset_support.pickle"
        self.re3 = re.compile(self.triple_dir + '/take_(\d\d)_(\d\d)_(\d\d).png')
        self.re2 = re.compile(self.double_dir + '/take_(\d\d)_(\d\d)_(\d\d).png')
        self.re1 = re.compile(self.single_dir + '/take_(\d\d)_(\d\d)_(\d\d).png')
        # do not change further

        self.saveSupport()
        print("DONE")

    def loadFileNames(self):
        r"""
        Load a list of all file names and creates the batches

        :returns: :class:`DataPreparator` current instance
        """
        self.file_list = glob(self.triple_dir + '/*.png')
        self.header["total"] = len(self.file_list)
        self.file_list = random.sample(self.file_list, self.header["total"])
        self.batches_fn = []
        for i in range(0, self.header["total"], self.header["batch_size"]):
            self.batches_fn.append(self.file_list[i:i + self.header["batch_size"]])
        self.header["batches"] = len(self.batches_fn)
        return self

    def saveBatches(self):
        r"""
        Saves batches in defined pickle file

        :returns: :class:`DataPreparator` current instance
        """
        self.loadFileNames()

        with open(self.out_file, "wb") as pickle_out:
            print("Dumping header")
            pickle.dump(self.header, pickle_out, pickle.HIGHEST_PROTOCOL)

            for no, batch_fn in enumerate(self.batches_fn):
                to_write = {"ids": [], "data": self.emptyBatch()}
                print("Working on batch {} of {}".format(no, self.header["batches"]))
                for i, file_name in enumerate(batch_fn):
                    to_write["data"][i, :, :, :] = self.loadFile(file_name)
                    to_write["ids"].append(self.re3.match(file_name).groups())
                pickle.dump(to_write, pickle_out, pickle.HIGHEST_PROTOCOL)
        return self

    def saveSupport(self):
        r"""
        Works on support files.

        :warning: this is one of those classes that is not possible to use a different data set form
        the one we are currently using

        :returns: :class:`DataPreparator`
        """
        single_files = glob(self.single_dir + "/*.png")
        double_files = glob(self.double_dir + "/*.png")
        data = {}
        with open(self.support_file, "wb") as f:
            print("Working on single files")
            for i in single_files:
                data[self.re1.match(i).groups()] = self.loadFile(i)
            print("Working on double files")
            for i in double_files:
                data[self.re2.match(i).groups()] = self.loadFile(i)
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return self

    def emptyBatch(self):
        r"""
        Create the empty batch to which it is possible to add data.

        :returns: numpy.ndarray
        """
        shape = (self.header["batch_size"], self.header["shape"][0], self.header["shape"][1], self.header["shape"][2])
        return np.ndarray(shape=shape, dtype=np.float32)

    def loadFile(self, f):
        r"""
        Loads the file and performs tra transformation in float values through this operation:

        $$
        p_{i,j} = \\dfrac{p_{i,j} - 1/2\\,(\\mathrm{max}(p) - \\mathrm{min}(p))}{(\\mathrm{max}(p) - \\mathrm{min}(p))}
        $$

        At the end the final image is reshaped to be a three dimensional tensor.

        :param f: filename
        :type f: str
        :returns: numpy.ndarray
        :raises: AssertionError, IOError
        """
        assert type(f) is str, "filename must be a str"
        try:
            image = ndimage.imread(f)
            image = (image - (image.max() - image.min()) / 2) / (image.max() - image.min())
            image = image.reshape(self.header["shape"])
            return image
        except IOError as e:
            print("Cannot read %s: %s. Skipping it!!" % (f, e))
            return None


#  ___       _          _  _              _ _
# |   \ __ _| |_ __ _  | || |__ _ _ _  __| | |___ _ _
# | |) / _` |  _/ _` | | __ / _` | ' \/ _` | / -_) '_|
# |___/\__,_|\__\__,_| |_||_\__,_|_||_\__,_|_\___|_|
class DataHandler:
    r"""
    This class will be used to handle the data to be passed to the convolutional autoencoder.
    The data model respond to the following rules:

     * there are two different files: the first one will contain all the images for the training
       with all the elements, while the second will contain only the factorized version of the images
     * since the training file create a too big memory footprint, each macro-batch is loaded one at the time.
     * the first reading in the complete file is an header that contains some information about the structure
       of the data. The dict contains a ``desc`` field with a description of each key meaning
     * after the first reading, each new reading will load a batch of element

    Actually nothing more than the initializer is needed to use this particular class::

         >>> for data in DataHandler(data_file, support_file, batch):
         ...     do_something(data)

    :warning: as for now this procedure is locked with only three elements due to short time.
    """
    def __init__(self, cf, sf):
        r"""
        Initialize the data handler and all the files

        :param cf: this is the file that contains the dataset
        :param sf: this is the support file
        :type cf: str
        :type sf: str
        :raises: AssertionError, IOError, RuntimeError
        :returns: :class:`DataHandler` new instance
        """
        assert type(cf) is str, "dataset filename must be a str"
        assert type(sf) is str, "support filename must be a str"
        assert isfile(cf), "dataset file does not exists"
        assert isfile(sf), "support file does not exists"

        with open(sf, "rb") as f:
            self.support = pickle.load(f)
        self.cf = cf

    def loop(self, b_msz=10, limit_batch=-1):
        r"""
        Executes the loop

        :param b_msz: this is the fine tuning dimension of the batch
        :param limit_batch: how many batches to run? if negative goes till the end
        :type b_msz: int
        :type limit_batch: int
        """
        assert type(b_msz) is int, "batch tuning size must be an int"
        assert type(limit_batch) is int, "limit batch must be an integer"
        assert b_msz > 0, "batch tuning size must be a positive number"
        with open(self.cf, "rb") as f:
            self.config = pickle.load(f)
            if limit_batch < 1 or limit_batch > self.config["batches"]:
                limit_batch = self.config["batches"]
            for batch_no in range(0, limit_batch):
                current = pickle.load(f)
                for i in range(0, np.shape(current["data"])[0] // b_msz):
                    init = i * b_msz
                    ends = init + b_msz
                    yield self._buildElement(
                        current["data"][init:ends, :, :, :],
                        current["ids"][init:ends]
                    )
        return None

    def _buildElement(self, cur_ndarray, cur_reference):
        r"""
        Build the array of data that should be sent as a block to the current batch to be trained
        As input ve have ``cur_ndarray`` that is the actual data, whilst in ``cur_refernce`` we have the
        reference for each element of the current refined batch::

            {im  im  im  im  im  ...} is numpy.ndarray
            [ref ref ref ref ref ...] is lst

        for each im we have a reference ``(x, y, z)`` that will be used to create the further
        references::

            (x, y, z) => (x, _, _), (_, y, _), (_, _, z), (x, y, _), (x, _, z), (_, y, z)

        each one of this new reference will create a new ``numpy.ndarray`` that is compatible with
        our stacked architecture
        """
        ret = [cur_ndarray]
        for i in range(0, 6):
            ret.append(np.ndarray(shape=np.shape(cur_ndarray), dtype=np.float32))
        for r in range(0, np.shape(cur_ndarray)[0]):
            for i, perm in self._permutations(cur_reference[r]):
                ret[i + 1][r, :, :, :] = self.support[perm]
        return ret

    def _permutations(self, l):
        assert type(l) is tuple, "input must be a tuple"
        perm = [
            (l[0], "99", "99"),
            ("99", l[1], "99"),
            ("99", "99", l[2]),
            (l[0], l[1], "99"),
            (l[0], "99", l[2]),
            ("99", l[1], l[2])
        ]
        for i, p in enumerate(perm):
            yield i, p
