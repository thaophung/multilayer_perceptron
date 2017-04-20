from __future__ import print_function
#import ipdb
#from util import print_equation
from cppn import build_cppn

import tensorflow as tf
import numpy as np
import itertools
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import pickle
#pickle.dump(mnist,open('mnist.p','wb'))

img1 = open('../make_datasets/reversal_train_inputs.pkl','rb')
training_images = pickle.load(img1)
label1 = open('../make_datasets/reversal_train_labels.pkl','rb')
training_labels = pickle.load(label1)

img2 = open('../make_datasets/reversal_test_inputs.pkl','rb')
testing_images = pickle.load(img2)
label2 = open('../make_datasets/reversal_test_labels.pkl','rb')
testing_labels = pickle.load(label2)

img1.close()
img2.close()
label1.close()
label2.close()

#for i in range(len(training_images)):
#    print_equation(training_images[i], training_labels[i])
#
#print ("----------")
#for i in range(len(testing_images)):
#    print_equation(testing_images[i], testing_labels[i])

# Parameter
learning_rate = 0.1
training_epochs = 100
batch_size = 25 
display_step = training_epochs / 10

beta = 0

# Network Parmeters
n_inputs = 15   # MNIST data input(img shape: 28x28)   # 22
n_hidden = 50
n_outputs = 15  # MNIST total classes (0-9 digits)     # 19

checkpoint_path = "./snapshots/model"

# Random seed
SEED = 0
tf.set_random_seed(SEED)
np.random.seed(SEED)

# tf Graph input
x = tf.placeholder("float", [None, n_inputs])
y = tf.placeholder("float", [None, n_outputs])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Output layer with linear activation
    h1 = tf.add( tf.matmul(x, weights['h1']), biases['h1'] )
    h1 = tf.nn.relu(h1)

    out = tf.add( tf.matmul(h1, weights['out']), biases['out'] )
    out = tf.nn.sigmoid(out)
    return out

# Store layers weight and bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_inputs, n_hidden])),
    'out': tf.Variable(tf.truncated_normal([n_hidden, n_outputs]))
}
biases = {
    'h1': tf.Variable(tf.truncated_normal([n_hidden])),
    'out': tf.Variable(tf.truncated_normal([n_outputs]))
}

# Construct main NN
pred = multilayer_perceptron(x, weights, biases)

# Construct CPPN
cppn, train_cppn_op, cppn_inputs, cppn_outputs = build_cppn()

# Use pickle to save any object to file/disk
# Define normal loss 
cost = tf.reduce_mean(tf.squared_difference(pred, y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Loss function with L2 Regularization with beta = 0.01
regularizers = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out'])
cost = tf.reduce_mean(cost + beta * regularizers)

# Define optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#grad = optimizer.compute_gradients(cost)
#train_op = optimizer.apply_gradients(grad)

def generate_coord( width, height ):
    return np.array(list(itertools.product(np.linspace(0,1, width), np.linspace(0,1, height)))).reshape(width * height, 2)

def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def save_model(saver, sess, checkpoint_path, epoch):
    """ saves the model to a file """
    saver.save(sess, checkpoint_path, global_step = epoch)

def load_model(saver, sess, checkpoint_path):

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print ("loading model: ",ckpt.model_checkpoint_path)

    #self.saver.restore(self.sess, checkpoint_path+'/'+ckpt.model_checkpoint_path)
    # use the below line for tensorflow 0.7
    saver.restore(sess, ckpt.model_checkpoint_path)

# Saver

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

    n_batch = int(training_images.shape[0]/batch_size)

    print ("Total examples", training_images.shape)
    print ("Batch size", batch_size)
    print ("Number of batches", n_batch)

    # Training cycle
    for epoch in range (training_epochs):
        avg_cost = 0.

        # Loop over all batches
        for i in range(n_batch):

            start = i * batch_size
            end = (i+1) * batch_size

#            print ( start, " ---> ", end )

            batch_x = training_images[ start:end, : ]
            batch_y = training_labels[ start:end, : ]

            # Backprop the error for the task to NN
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            #Compute average loss
            avg_cost += c/n_batch
    
            # Get the gradient and update CPPN
        
            '''
            # Generate weight matrix via the updated CPPN
            _, generated_weights = sess.run([train_cppn_op, cppn], 
                        feed_dict={
                                cppn_inputs: generate_coord(width = n_inputs, height = n_outputs),
                                cppn_outputs: tf.reshape( weights['out'], ( n_inputs * n_outputs, 1)).eval() 
                            })
            generated_weights = generated_weights.reshape((n_inputs, n_outputs, 2))

            # Update the main NN with generated weights
            update_weights = tf.assign(weights['out'], generated_weights[:,:,0])
            update_biases = tf.assign(biases['out'], generated_weights[0,:,1])
            sess.run( [update_weights, update_biases] )

            '''

        #Display logs per epoch step
        if epoch % display_step == 0:

            # Test model
            correct_prediction = tf.equal(pred, y)

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            train_accuracy = accuracy.eval({x:training_images, y: training_labels})
            test_accuracy = accuracy.eval({x:testing_images, y: testing_labels})
#            print ("Training accuracy:", accuracy.eval({x:training_images, y: training_labels}))
#            print("Test accuracy:", accuracy.eval({x:testing_images, y: testing_labels}))

            print("Epoch:", '%04d' % (epoch+1), \
                        "cost=", "{:.9f}".format(avg_cost), \
                        "train=", "{:.4f}".format(train_accuracy), \
                        "test=", "{:.4f}".format(test_accuracy)) 

            save_model(saver, sess, checkpoint_path, epoch)

    print ("Optimization Finish!")

