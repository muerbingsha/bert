import numpy as np
import tensorflow as tf
import time, os
import urllib.request
import matplotlib.pyplot as plt


class CustomeCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, num_weights):
        self.num_units = num_units
        self.num_weights = num_weights

    @property
    def state_size(self):
        return self.num_units
