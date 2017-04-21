import numpy as np
import random
import pickle


def create_data(batch_size):
    inputs = np.zeros((batch_size,22))
    labels = np.zeros((batch_size,1))
    #for i in range(0,batch_size-1):
    i = 0
    while i < batch_size:
        j = 0
        # Create 1+2
        for j in range(0,1):
            inputs[j+i,1] = 1
            inputs[j+i,10] = 1
            inputs[j+i,20] = 1
            labels[j+i,0] = 9
            j += 1
        # 2+1
        for j in range(1,2):
            inputs[j+i,8] = 1
            inputs[j+i,10] = 1
            inputs[j+i,13] =1
            labels[j+i, 0] = 9
            j+= 1
        # Create 0+
        i = i+2

    image_output = open("math.testing.images", "wb")
    pickle.dump(inputs,image_output)
    
    label_output = open("math.testing.labels", "wb")
    pickle.dump(labels, label_output)
create_data(2)
