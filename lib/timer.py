#!/usr/bin/env python3

import time


def timestring():
    """
    Returns a timestring formatted as: ``YYYYMMDD-HHmm``, used for automatic folder generation

    :returns: str
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M")


class Timer(object):
    """
    Helper class for timing (tic-toc and global). The first time the class is called inside a
    script, a tic timing is taken, to be used as absolute reference.

    The timing of a portion of the code is obtained in a context, as follows::

        >>> with Timer():
        ...    do_something()
        ...
        Time context: 0:02:20.934
        Time total: 0:02:20.934
    """
    start = None

    def __init__(self):
        self.tic = None
        self.toc = None
        if Timer.start is None:
            Timer.start = time.time()

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, type, value, traceback):
        self.toc = time.time()
        print(self)

    def __str__(self):
        local_diff = self.toc - self.tic
        local_h, local_m, local_s = self.__split_time__(local_diff)
        total_diff = self.toc - Timer.start
        total_h, total_m, total_s = self.__split_time__(total_diff)
        return ("Time context: %d:%02d:%02.3f\n" % (local_h, local_m, local_s)) + ("Time total: %d:%02d:%02.3f" % (total_h, total_m, total_s))

    def __split_time__(self, local_diff):
        local_h = local_diff // (60 * 60)
        local_m = (local_diff - local_h * 60 * 60) // 60
        local_s = (local_diff - local_h * 60 * 60) - local_m * 60
        return [local_h, local_m, local_s]
