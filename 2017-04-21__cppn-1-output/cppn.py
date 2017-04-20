import tensorflow as tf
#import ipdb

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def build (X, w):
    h = tf.nn.relu(tf.matmul(X, w['h1']))
    h = tf.nn.relu(tf.matmul(h, w['h2']))
    output = tf.matmul(h, w['out'])
    return output

#def generate_coord( width, height ):
#    return np.array(list(itertools.product(np.linspace(0,1, width), np.linspace(0,1, height)))).reshape(width * height, 2)

def build_cppn (L2_weight):
    cppn_inputs = 2
    cppn_outputs = 2
    cppn_units = 20 

#    generate_coord(width = n_inputs, height = n_outputs),
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

        regularizers = tf.nn.l2_loss(w['h1']) + tf.nn.l2_loss(w['h2']) + tf.nn.l2_loss(w['out']) 
        cost = tf.reduce_mean(L2_weight * regularizers)

    return cppn, cost, X, Y
