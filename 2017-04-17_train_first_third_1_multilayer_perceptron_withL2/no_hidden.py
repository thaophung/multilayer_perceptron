from __future__ import print_function
import ipdb
from util import print_equation
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

import tensorflow as tf
import numpy as np

# Parameter
learning_rate = 0.1
training_epochs = 10000
batch_size = 47
display_step = 1000

beta=0.0005
# Network Parmeters
n_input = 22    # MNIST data input(img shape: 28x28)   # 22
n_classes = 19  # MNIST total classes (0-9 digits)     # 19

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Output layer with linear activation
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

# Store layers weight and bias
weights = {
        'out': tf.Variable(tf.random_normal([n_input, n_classes]))
}
biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Use pickle to save any object to file/disk
# Define normal loss 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Loss function with L2 Regularization with beta = 0.01
#regularizers = tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(weights['out'])
#cost = tf.reduce_mean(cost + beta * regularizers)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

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
        #for i in range(total_batch):
#            batch_x, batch_y = mnist.train.next_batch(batch_size)
            #batch_x, batch_y = next_batch(batch_size)
#            ipdb.set_trace()
            # Run optimization op (backprop) and cost op (to get loss value)
        i=0
        while i < total_batch:
            #ipdb.set_trace()
            batch_x=training_images[i:(i+batch_size),:]
           # print batch_x
            batch_y= training_labels[i:(i+batch_size),:]
           # print batch_y
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                        y: batch_y})
            #Compute average loss
            avg_cost += c/total_batch
    
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

