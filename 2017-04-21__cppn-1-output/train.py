'''
Anh Nguyen <anh.ng8@gmail.com> 2017
'''

import itertools

import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform
import sys
import ntpath
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7
#import ipdb

"""User Parameters"""
# The train image will be scaled to a square of dimensions `train_size x train_size`
train_size = 32
# When generating the image, the network will generate for an image of
# size `test_size x test_size`
test_sizes = [ 32, 64, 2048 ]
# Path to load the image you want upscaled
#image_path = '../img/colors.jpg'
image_path = sys.argv[1] #"img/colors.jpg"
filename = ntpath.basename(image_path)

###############################
# USAGE
# CUDA_VISIBLE_DEVICES=2 python train.py images/colors.jpg
###############################

SEED = 0

if not image_path:
    print('Please specify an image for training the network')
else:
    image = transform.resize(io.imread(image_path), (train_size, train_size))
    # Just a quick line to get rid of the alpha channel if it exists
    # (e.g. for transparent png files)
    image = image if len(image.shape) < 3 or image.shape[2] == 3 else image[:,:,:3]
    print "Resized to ", image.shape
#    io.imshow(image)


def model(X, w):
    h = tf.sin(tf.matmul(X, w['h1']))
    h = tf.sin(tf.matmul(h, w['h2']))
    h = tf.sin(tf.matmul(h, w['h3']))
    h = tf.sin(tf.matmul(h, w['h4']))
    h = tf.sin(tf.matmul(h, w['h5']))
    h = tf.sin(tf.matmul(h, w['h6']))
    h = tf.sin(tf.matmul(h, w['h7']))
    h = tf.sin(tf.matmul(h, w['h8']))    
    return tf.nn.sigmoid(tf.matmul(h, w['out']))


def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def save_model(saver, sess, checkpoint_path, epoch):
    """ saves the model to a file """
    saver.save(sess, checkpoint_path, global_step = epoch)

def load_model(saver, sess, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print "loading model: ",ckpt.model_checkpoint_path

    #self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
    # use the below line for tensorflow 0.7
    saver.restore(sess, ckpt.model_checkpoint_path)


nb_inputs = 2   # x, y
nb_outputs = 3  # R, G, B
nb_units = 20 # per layer

checkpoint_path = "./snapshots/cppn"

# Random seed
tf.set_random_seed(SEED)
np.random.seed(SEED)

# Inputs are coordinates
X = tf.placeholder('float32', (None, nb_inputs)) 

# (None, None) refers to (batch_size, n_colors)
Y = tf.placeholder("float32", (None, None))

w = {
  'h1': init_weights([nb_inputs, nb_units]),
  'h2': init_weights([nb_units, nb_units]),
  'h3': init_weights([nb_units, nb_units]),
  'h4': init_weights([nb_units, nb_units]),
  'h5': init_weights([nb_units, nb_units]),
  'h6': init_weights([nb_units, nb_units]),
  'h7': init_weights([nb_units, nb_units]),
  'h8': init_weights([nb_units, nb_units]),
  'out': init_weights([nb_units, nb_outputs]),
}

# Construct the graph
out = model(X, w)

# MSE loss
cost = tf.reduce_mean(tf.squared_difference(out, Y))
train_op = tf.train.AdamOptimizer().minimize(cost)

global_step = tf.Variable(0, trainable=False)
increment_global_step_op = tf.assign(global_step, global_step+1)

#starter_learning_rate = 0.001
#learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,
#                                           decay_steps=2000, decay_rate=0.6, staircase=True)
# Passing global_step to minimize() will increment it at each step.
# Use simple momentum for the optimization.
#train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)

#train_op = tf.train.exponential_decay(learning_rate=0.01, lr_decay=0.96).minimize(cost)
#train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01, lr_decay=0.96).minimize(cost)
#train_op = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.96, momentum=0.9, epsilon=1e-12).minimize(cost)
# Feel free to adjust the number of epochs to your liking.
n_epochs = 5e+4

# Create function to generate a coordinate matrix (i.e. matrix of normalised coordinates)
# Pardon my lambda 
generate_coord = lambda size: (
    np.array(list(itertools.product(np.linspace(0,1,size),np.linspace(0,1,size)))).reshape(size ** 2, 2))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
    
    # Training data
    x = generate_coord(train_size)

    # Labels
    reshaped_image = np.array(image.reshape(train_size ** 2, -1))
    
    for epoch in range(int(n_epochs + 1)):
        # Update model
        _, _, c = sess.run([train_op, increment_global_step_op, cost], feed_dict={X: x, Y: reshaped_image})
 
        # Print progress
        if epoch % (n_epochs/10) == 0:
#        if epoch % 100 == 0:
            print('{:0.0%} \t Loss: {}'.format(epoch/n_epochs, c).expandtabs(7))
            save_model(saver, sess, checkpoint_path, epoch)
    
    for test_size in test_sizes:
        # Generate an image
        generated_image = sess.run(out, feed_dict={X: generate_coord(test_size)})
        generated_image = generated_image.reshape((test_size, test_size, 3))

        if train_size != test_size:
            resized_image = transform.resize(image, (test_size, test_size))
        else:
            resized_image = image.copy()

        print_image = np.concatenate((resized_image, generated_image), axis=1)
      
        path = "%s_%04d.jpg" % (filename, test_size)
        io.imsave(path, print_image)
        print "Saved to [ %s ]" % path
