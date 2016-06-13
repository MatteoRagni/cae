#!/usr/bin/env python3

"""
This module implements a really simple Bot notifier on Telegram service.
The configuration file must be in: `/etc/notify-telegram.conf`
"""

import requests
from os.path import isfile
import yaml


class string(str):
    """
    String splitter based on length
    """
    def chunk(self, length):
        assert type(length) is int, "Chunk lenght must be an int. Received a %s" % type(length)
        assert length > 0, "Chunk lenght must be positive. Received %d" % length
        for i in range(0, len(self), length):
            yield self[i:i + length]


class TelegramNotifier:
    CONFIG_PATH = '/etc/notify-telegram.conf'
    MAX_LENGTH  = 3000

    def __init__(self):
        """
        Initialize the telegram notifier object using options defined in the configuration
        YAML file: `/etc/notify-telegram.conf`
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
        Post the message `m` that must be a string not empty (`m != ""`)
        """
        if self.config is None:
            pass
        assert type(m) is str, "Message must be a string"
        assert m == "", "Message must be string not empty"
        payload = {'chat_id': self.chatid, 'text': ""}
        for x in string(m).chunk(self.MAX_LENGTH):
            payload['text'] = x
            requests.post(self.uri_path, payload)
