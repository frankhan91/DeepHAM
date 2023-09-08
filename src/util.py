from tensorflow import keras
import tensorflow as tf
import numpy as np

# def create_model(d_in, d_out, config):
#     model = keras.Sequential()
#     model.add(keras.layers.InputLayer([d_in]))
#     for w in config["net_width"]:
#         model.add(keras.layers.Dense(w, activation=config["activation"]))
#     model.add((keras.layers.Dense(1, activation=None)))
#     return model

class FeedforwardModel(keras.Model):
    def __init__(self, d_in, d_out, config, name="agentmodel", **kwargs):
        super(FeedforwardModel, self).__init__(name=name, **kwargs)
        self.dense_layers = [keras.layers.Dense(w, activation=config["activation"]) for w in config["net_width"]]
        self.dense_layers.append(keras.layers.Dense(d_out, activation=None))
        self.d_in = d_in

    def call(self, inputs):
        x = self.dense_layers[0](inputs)
        for l in self.dense_layers[1:]:
            x = l(x)
        return x

    def load_weights_after_init(self, path):
        # evaluate once for creating variables before loading weights
        zeros = tf.ones([1, 1, self.d_in])
        self.__call__(zeros)
        self.load_weights(path)

class GeneralizedMomModel(FeedforwardModel):
    def __init__(self, d_in, d_out, config, name="generalizedmomentmodel", **kwargs):
        super(GeneralizedMomModel, self).__init__(d_in, d_out, config, name=name, **kwargs)

    def basis_fn(self, inputs):
        x = self.dense_layers[0](inputs)
        for l in self.dense_layers[1:]:
            x = l(x)
        return x

    def call(self, inputs):
        x = self.basis_fn(inputs)
        gm = tf.reduce_mean(x, axis=-2, keepdims=True)
        gm = tf.tile(gm, [1, inputs.shape[-2], 1])
        return gm

def print_elapsedtime(delta):
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def gini(array): #https://github.com/oliviaguest/gini
    """Calculate the Gini of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() # all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) # values cannot be negative
    array += 0.0000001 # values cannot be 0
    array = np.sort(array) # values must be sorted
    index = np.arange(1, array.shape[0]+1) # index per array element
    n = array.shape[0] # number of array elements
    return (np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)) # Gini coefficient
