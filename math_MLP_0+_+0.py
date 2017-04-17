from __future__ import print_function
import ipdb
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

import tensorflow as tf
import numpy as np
#import ipdb

# Parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 5 # 1st layer number of features
n_hidden_2 = 5 # 2nd layer number of features
n_input = 22 # MNIST data input(img shape: 28x28)   # 22
n_classes = 19  # MNIST total classes (0-9 digits)      # 19

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight and bias
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # MNIST: 60K training images
    # 1 epoch: ---> 60K images
    # 1 epoch: batch_size = 100, num of batches: 600


    # Training cycle
    for epoch in range (training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        total_batch = 76000
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
    
            i +=batch_size
        #Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
    print ("Optimization Finish!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print ("Training accuracy:", accuracy.eval({x:training_images, y: training_labels}))
    print("Test accuracy:", accuracy.eval({x:testing_images, y: testing_labels}))

    img1.close()
    img2.close()
    label1.close()
    label2.close()
