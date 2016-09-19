#! /usr/bin/env python3

from six.moves import cPickle as pickle
import cmd
from os.path import isfile
from autoencoder import ConvAutoEncSettings

def print_warning(msg):
    assert type(msg) is str, "A string is required"
    print('\033[91m' + msg + '\033[0m')

def print_error(msg):
    assert type(msg) is str, "A string is required"
    print('\033[93m' + msg + '\033[0m')

def print_info(msg):
    assert type(msg) is str, "A string is required"
    print('\033[92m' + msg + '\033[0m')

class SettingsInterface(cmd.Cmd):
    prompt = "[??] > "
    def do_load(self, arg):
        """
        Load a configuration from a pickle file.

          load FILENAME
        """
        try:
            if type(arg) is str:
                if isfile(arg):
                    try:
                        with open(arg, "rb") as fp:
                            self.data = pickle.load(fp)

                            if type(self.data) is tuple:
                                for cae in self.data:
                                    if type(cae) is not ConvAutoEncSettings:
                                        self.data = None
                                        print_error("File does not contain ConvAutoEncSettings elements")
                                        return
                                self.prompt = "[{}] > ".format(arg)
                                print_info("Loaded {}".format(arg))
                                print_info("File contains {} blocks".format(len(self.data)))
                            else:
                                self.data = None
                    except TypeError:
                        print_error("Cannot load file")
                else:
                    print_warning("File not found")
            else:
                print_error("Argument of load must be a str")
        except Exception as e:
            print_error("Error: {}".format(e))


    def do_save(self, arg):
        try:
            if type(arg) is str:
                with open(arg, "wb") as fp:
                    pickle.dump(self.data, fp)
                self.prompt = "[{}] > ".format(arg)
        except Exception as e:
            print_error("Error: {}".format(e))

    def do_layer_new(self, args):
        try:
            pass
        except Exception as e:
            print_error("Error: {}".format(e))

    def do_autoencoder_new(self, args):
        try:
            pass
        except Exception as e:
            print_error("Error: {}".format(e))

    def do_layer_set(self, args):
        try:
            pass
        except Exception as e:
            print_error("Error: {}".format(e))

    def do_autoencoder_set(self, args):
        try:
            pass
        except Exception as e:
            print_error("Error: {}".format(e))


if __name__ == '__main__':
    SettingsInterface().cmdloop()
