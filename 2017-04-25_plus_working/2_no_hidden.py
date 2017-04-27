from __future__ import print_function
#import ipdb
from util import print_equation
#from cppn import build_cppn

import tensorflow as tf
import numpy as np
import itertools

import pickle

# Load training and test set
inputs1 = open('training_inputs.pkl', 'rb')
training_inputs = pickle.load(inputs1)

labels1 = open('training_labels.pkl', 'rb')
training_labels = pickle.load(labels1)

inputs2 = open('test_inputs.pkl', 'rb')
test_inputs = pickle.load(inputs2)

labels2 = open('test_labels.pkl', 'rb')
test_labels = pickle.load(labels2)

inputs1.close()
inputs2.close()
labels1.close()
labels2.close()

# Print the training and test set
for i in range(len(training_inputs)):
    print_equation(training_inputs[i], training_labels[i])
print("--------------------")

for i in range(len(test_inputs)):
    print_equation(test_inputs[i], test_labels[i])
#-------------------

# Parameter
learning_rate = 0.1
training_epochs = 20000
batch_size = 22
display_step = training_epochs/10
test_step = training_epochs/100

beta = 0.005

# Network Parameters
n_inputs = 22
n_hidden = 10       
n_outputs = 1

# Random seed
SEED = 0    
tf.set_random_seed(SEED)
np.random.seed(SEED)

# tf Graph input
x = tf.placeholder("float", [None, n_inputs])
y = tf.placeholder("float", [None, n_outputs])

# Create model 
def multilayer_perceptron(x, weights, biases):
    #Output layer with linear activation
    h1 = tf.matmul(x,weights['h1']) + biases['h1']      
    h1 = tf.nn.relu(h1)

    out_layer = tf.matmul(h1, weights['out']) + biases['out']
    return out_layer

#  Store layers weight and bias
weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
}

biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_outputs]))
}

# Construct main NN
pred = multilayer_perceptron(x, weights, biases)

# Calculate loss and L2 regularizer
# MSE
cost = tf.reduce_mean(tf.squared_difference(pred, y))

regularizers = tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(weights['h1'])
cost = tf.reduce_mean(cost + beta * regularizers)

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

def generate_coord(width, height):
    return np.array(list(itertools.product(np.linspace(0,1,width), np.linspace(0,1,height)))).reshape(width * height, 2)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    prev_train_accuracy = 0
    prev_test_accuracy = 0

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 1

        i = 0

        while i < total_batch:
            batch_x = training_inputs[i:(i+batch_size)]
            batch_y = training_labels[i:(i+batch_size)]

            # Backprop the error for the task to NN
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

            # Compute average loss
            avg_cost += c/total_batch

            i+= batch_size

        if epoch % test_step == 0:
            # Test model
            correct_prediction = tf.equal(tf.cast(pred, dtype=tf.int32), tf.cast(y, dtype=tf.int32))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            train_accuracy = accuracy.eval({x:training_inputs, y:training_labels})
            test_accuracy = accuracy.eval({x:test_inputs, y:test_labels})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                    "cost=", "{:.9f}".format(avg_cost), \
                    "train=", "{:.4f}".format(train_accuracy), \
                    "test=", "{:.4f}".format(test_accuracy), \
                    "best train=", "{:.4f}".format(prev_train_accuracy), \
                    "best test=", "{:.4f}".format(prev_test_accuracy))

            for j in [0, 5, 10, 15]:
                test_eq = training_inputs[j]
                test_answer = pred.eval({x:test_eq.reshape(1,22)})[0]
                #print_equation(test_eq.toList(), test_answer.toList())
                print_equation(test_eq, test_answer)

            print ("------Heldout equations--------")
            for j in range(len(test_inputs)):
                test_eq = test_inputs[j]
                test_answer = pred.eval({x:test_eq.reshape(1,22)})[0]
                #print_equation(test_eq.toList(), test_answer.toList())
                print_equation(test_eq, test_answer)

        if train_accuracy > prev_train_accuracy:
            prev_train_accuracy = train_accuracy
        if test_accuracy > prev_test_accuracy:
            prev_test_accuracy = test_accuracy

        if prev_train_accuracy == 1.0 and prev_test_accuracy == 1.0:
            print ("Epoch:", '%04d' % (epoch+1), \
                    "cost=", "{:.9f}".format(avg_cost), \
                    "train=", "{:.4f}".format(train_accuracy), \
                    "test=", "{:.4f}".format(test_accuracy), \
                    "best train=", "{:.4f}".format(prev_train_accuracy), \
                    "best test=", "{:.4f}".format(prev_test_accuracy))

            for j in [0,5,10,15]:
                test_eq = training_inputs[j]
                test_answer = pred.eval({x:test_eq.reshape(1,22)})[0]
                print_equation(test_eq, test_answer)

            print ("------Heldout equations--------")
            for j in range(len(test_inputs)):
                test_eq = test_inputs[j]
                test_answer = pred.eval({x:test_eq.reshape(1,22)})[0]
                print_equation(test_eq, test_answer)
            break
    print("Optimization Finish!")
