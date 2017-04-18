import tensorflow as tf
#import ipdb

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def build (X, w):
    h = tf.nn.relu(tf.matmul(X, w['cppn_h1']))
    h = tf.nn.relu(tf.matmul(h, w['cppn_h2']))
    output = tf.matmul(h, w['cppn_out'])
    return output

def build_cppn ():
    cppn_inputs = 2
    cppn_outputs = 2
    cppn_units = 20

    with tf.name_scope("cppn"):

        # Inputs are coordinates
        X = tf.placeholder('float32', (None, cppn_inputs))

        # (None, None) refers to (batch_size, n_colors)
    #    Y = tf.placeholder("float32", (None, cppn_outputs))

        w = {
          'cppn_h1' : init_weights([cppn_inputs, cppn_units]),
          'cppn_h2' : init_weights([cppn_units, cppn_units]),
          'cppn_out': init_weights([cppn_units, cppn_outputs]),
        }

        cppn = build (X, w)

    return cppn, X
