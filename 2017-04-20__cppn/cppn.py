import tensorflow as tf
#import ipdb

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def build (X, w):
    h = tf.sin(tf.matmul(X, w['h1']))
    h = tf.sin(tf.matmul(h, w['h2']))
    output = tf.matmul(h, w['out'])
    return output

def build_cppn ():
    cppn_inputs = 2
    cppn_outputs = 2
    cppn_units = 10 

    with tf.name_scope("cppn"):

        # Inputs are coordinates
        X = tf.placeholder('float32', (None, cppn_inputs))

        # (None, None) refers to (batch_size, n_colors)
        Y = tf.placeholder("float32", (None, None))

        w = {
          'h1' : init_weights([cppn_inputs, cppn_units]),
          'h2' : init_weights([cppn_units, cppn_units]),
          'out': init_weights([cppn_units, cppn_outputs]),
        }

        cppn = build (X, w)


    # Define MSE loss function for CPPN
    cost = tf.reduce_mean(tf.squared_difference(cppn, Y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

    return cppn, train_op, X, Y
