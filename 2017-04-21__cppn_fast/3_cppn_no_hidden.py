from __future__ import print_function
#import ipdb
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

for i in range(len(training_images)):
    print_equation(training_images[i], training_labels[i])

print ("----------")
for i in range(len(testing_images)):
    print_equation(testing_images[i], testing_labels[i])

# Parameter
learning_rate = 0.1
training_epochs = 1000
batch_size = 47
display_step = training_epochs / 10

L2_weight = 0.005

# Network Parmeters
n_inputs = 22    # MNIST data input(img shape: 28x28)   # 22
n_outputs = 19   # MNIST total classes (0-9 digits)     # 19

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
    out_layer = tf.matmul(x, weights) + biases

#    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

# Construct CPPN
cppn, cppn_loss, cppn_inputs, cppn_outputs = build_cppn(L2_weight=L2_weight)
cppn = tf.reshape( cppn, (n_inputs, n_outputs, 2))

W = cppn[:,:,0]
B = cppn[0,:,1]

# Construct main NN
pred = multilayer_perceptron(x, W, B)

# Use pickle to save any object to file/disk
# Define normal loss 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

def generate_coord( width, height ):
    return np.array(list(itertools.product(np.linspace(0,1, width), np.linspace(0,1, height)))).reshape(width * height, 2)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
#    for epoch in range (training_epochs):
    epoch = -1
    while True:
        epoch += 1

        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        total_batch = 1

        i=0

        coords = generate_coord(width = n_inputs, height = n_outputs)

        while i < total_batch:

            batch_x = training_images[i:(i+batch_size),:]
            batch_y = training_labels[i:(i+batch_size),:]

            # Backprop the error for the task to NN
            _, c = sess.run([optimizer, cost], 
                        feed_dict={
                            x: batch_x, 
                            y: batch_y, 
                            cppn_inputs: coords
                        })

            #Compute average loss
            avg_cost += c/total_batch

            i += batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            train_accuracy = accuracy.eval({x:training_images, y: training_labels, cppn_inputs: coords})
            test_accuracy = accuracy.eval({x:testing_images, y: testing_labels, cppn_inputs: coords})

            print("Epoch:", '%04d' % (epoch+1), \
                        "cost=", "{:.9f}".format(avg_cost), \
                        "train=", "{:.4f}".format(train_accuracy), \
                        "test=", "{:.4f}".format(test_accuracy)) 
#
#            for j in [ 10, 20, 30 ]:
#                test_eq = training_images[j]
#                test_answer = pred.eval({x: test_eq.reshape(1, 22)})[0]
#                print_equation(test_eq.tolist(), test_answer.tolist())
#
#            for j in range(testing_images.shape[0]):
#                test_eq = testing_images[j]
#                test_answer = pred.eval({x: test_eq.reshape(1, 22)})[0]
#                print_equation(test_eq.tolist(), test_answer.tolist())

    print ("Optimization Finish!")

