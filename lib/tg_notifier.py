#!/usr/bin/env python3

r"""
.. module:: tg_notifier
   :platform: Linux
   :synopsis: Simplest Telegram notifier ever

This module implements a really simple Bot notifier on Telegram service.
The configuration file must be in: `/etc/notify-telegram.conf`. It is used
in case of very long simulations.

.. :moduleauthor: Matteo Ragni
"""

import requests
from os.path import isfile
import yaml


class string(str):
    r"""
    Expansion of the :class:`str`.
    """
    def chunk(self, length):
        r"""
        String splitter based on length

        :param length: splitting length
        :param type: int
        :returns: :class:`string` current instance
        :raises: AssertionError
        """
        assert type(length) is int, "Chunk lenght must be an int. Received a %s" % type(length)
        assert length > 0, "Chunk lenght must be positive. Received %d" % length
        for i in range(0, len(self), length):
            yield self[i:i + length]
        return self


class TelegramNotifier:
    r"""
    A very simple notification tool on Telegram. It uses as configuration a YAML file that is
    usually stored in ``/etc/notify-telegram.conf``. The message are of maximum 4090 chars. Longer
    messages are automatically splitted in multiple messages.
    """
    CONFIG_PATH = '/etc/notify-telegram.conf'
    MAX_LENGTH  = 4090

    def __init__(self):
        """
        Initialize the telegram notifier object using options defined in the configuration
        YAML file: `/etc/notify-telegram.conf`

        If some `Exception` is raised, the will be loaded as dummy (:py:attr:`pass` an all methods).

        :returns: :class:`TelegramNotifier` new instance
        """
        try:
            assert isfile(self.CONFIG_PATH), "Configuration file not found in %s" % self.CONFIG_PATH
            with open(self.CONFIG_PATH, 'r') as f:
                self.config = yaml.load(f)
                self.uri_path = self.config['uri'] + self.config['token'] + '/sendMessage'
                self.chatid = self.config['chatid']
        except AssertionError as error:
            print("TELEGRAM_NOTIFIER: Configuration file not found. All methods will pass")
            print(error)
            self.config = None
            pass
        except yaml.YAMLError as error:
            print("TELEGRAM_NOTIFIER: Cannot read configuration yaml. All methods will pass")
            print(error)
            self.config = None
            pass
        except KeyError as error:
            print("TELEGRAM_NOTIFIER: Some key are missing. All methods will pass")
            print(error)
            self.config = None
            pass

    def post(self, m):
        """
        Post the message `m` that must be a string not empty (`m != ""`). If the message is longer
        than 4090 characters, than it will be splitted in more messages automatically

        :param m: message to be sent
        :type m: str
        :raises: AssertionError
        """
        if self.config is None:
            pass
        assert type(m) is str, "Message must be a string"
        assert m == "", "Message must be string not empty"
        payload = {'chat_id': self.chatid, 'text': ""}
        for x in string(m).chunk(self.MAX_LENGTH):
            payload['text'] = x
            requests.post(self.uri_path, payload)
