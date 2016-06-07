#!/usr/bin/env python3

import os
import numpy as np
import random

from six.moves import cPickle as pickle
from contextlib import contextmanager

# This version willl create two new object data. The first one is an
# object created only by intensity.
# The other one is the object created by the difference with respect to the mean


class DataHandler(object):

    def __init__(self, name):
        assert type(name) is str, "Name must be a string"
        assert os.path.isfile(name) is True, "File must exist"
        self.name = name

    def elaborate(self, limit=0):
        assert type(limit) is int, "Limits must be a "
        with open(self.name, "rb") as f:
            try:
                self.batches = pickle.load(f)
                assert type(self.batches) is int, "First read must be an integer (batches_no)"
                
