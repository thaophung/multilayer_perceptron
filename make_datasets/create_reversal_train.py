import numpy as np
import random
import pickle

def create_data(dataset_size, mod, name):
    size = 15 

    inputs = np.zeros((dataset_size, size))
    labels = np.zeros((dataset_size, size))

    idx = 0

    for i in range(0, dataset_size * 2):
        if i % 2 == mod:

            # Example
            s = '{0:015b}'.format(i)
            inputs[idx] = [ int(b) for b in s ]

            # Label
            labels[idx] = inputs[idx][::-1]

            idx += 1 
    
    print inputs[-1]
    print "-------------"
    print labels[-1]
    print "==============================="

    image_output = open("%s_inputs.pkl" % name, "wb")
    pickle.dump(inputs,image_output)
    
    label_output = open("%s_labels.pkl" % name, "wb")
    pickle.dump(labels, label_output)

SIZE = 25


create_data(dataset_size=SIZE, mod=0, name="reversal_train") 
create_data(dataset_size=SIZE, mod=1, name="reversal_test")
