from __future__ import print_function
import ipdb
from util import print_equation
from cppn import build_cppn

import tensorflow as tf
import numpy as np
import itertools
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import pickle
#pickle.dump(mnist,open('mnist.p','wb'))
img1 = open('math.training.images','rb')
training_images = pickle.load(img1)
label1 = open('math.training.labels','rb')
training_labels = pickle.load(label1)
img2 = open('math.testing.images','rb')
testing_images = pickle.load(img2)
label2 = open('math.testing.labels','rb')
testing_labels = pickle.load(label2)

img1.close()
img2.close()
label1.close()
label2.close()

#for i in range(len(training_images)):
#    print_equation(training_images[i], training_labels[i])
#ipdb.set_trace()


# Parameter
learning_rate = 0.1
training_epochs = 10000
batch_size = 47
display_step = 100

beta = 0.0005
# Network Parmeters
n_inputs = 22    # MNIST data input(img shape: 28x28)   # 22
n_outputs = 19   # MNIST total classes (0-9 digits)     # 19

# tf Graph input
x = tf.placeholder("float", [None, n_inputs])
y = tf.placeholder("float", [None, n_outputs])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Output layer with linear activation
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

# Store layers weight and bias
weights = {
    'out': tf.Variable(tf.random_normal([n_inputs, n_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_outputs]))
}

# Construct main NN
pred = multilayer_perceptron(x, weights, biases)

# Construct CPPN
cppn, train_cppn_op, cppn_inputs, cppn_outputs = build_cppn()

# Use pickle to save any object to file/disk
# Define normal loss 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Loss function with L2 Regularization with beta = 0.01
#regularizers = tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(weights['out'])
#cost = tf.reduce_mean(cost + beta * regularizers)

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(cost)

#optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(cost)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#grad = optimizer.compute_gradients(cost)
#train_op = optimizer.apply_gradients(grad)

def generate_coord( width, height ):
    return np.array(list(itertools.product(np.linspace(0,1, width), np.linspace(0,1, height)))).reshape(width * height, 2)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range (training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        total_batch = 1

        i=0

        while i < total_batch:

            batch_x = training_images[i:(i+batch_size),:]
            batch_y = training_labels[i:(i+batch_size),:]

            # Backprop the error for the task to NN
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            #Compute average loss
            avg_cost += c/total_batch
    
            # Get the gradient and update CPPN
#            sess.run(train_cppn_op, feed_dict={cppn_inputs: })
        
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

            i += batch_size

        #Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))

    print ("Optimization Finish!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Training accuracy:", accuracy.eval({x:training_images, y: training_labels}))
    print("Test accuracy:", accuracy.eval({x:testing_images, y: testing_labels}))
