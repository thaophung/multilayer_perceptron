import numpy as np
import random
import pickle


def create_data(batch_size):
    inputs = np.zeros((batch_size,22))
    labels = np.zeros((batch_size,19))
    #for i in range(0,batch_size-1):
    i = 0
    while i < batch_size:
        j = 0
        # Create first digit and blank
        for j in range(0,10):
            inputs[i,j] = 1
            labels[i,j] =1
            i += 1
            j += 1
        # Create blank and digit
        for j in range(10,20):
            inputs[i,j+12-10] = 1
            labels[i, j-10] = 1
            i+=1
            j += 1
        # Create 1+ without 1+2
        for j in range(20,30):
            if j == 28:             # If you want exlude 1+8, (if j == 28),  if you want exclude 1+3, (if j = 23)
                j+=1
                continue
            inputs[i,1] = 1
            inputs[i,10] = 1
            inputs[i,j-20+12] = 1
            labels[i,j-20+1] = 1
            j+= 1
            i+=1
        # Create +1 without 2+1
        for j in range(30,40):
            if j == 38:             # exclude 8+1 (if j == 38), exclude 3+1, (if j == 31), 
                j+=1
                continue
            inputs[i, j-30] = 1     
            inputs[i, 10] = 1
            inputs[i, 13] = 1
            labels[i,j- 30 +1] = 1
            i+=1
            j+=1
        
        # Create -1
        for j in range(40,50):
            if j == 40:
                j+=1
                continue
            inputs[i,j-40]=1
            inputs[i,11] = 1
            inputs[i, 13] = 1
            labels[i, j-40-1] = 1
            i+=1
            j+=1

    # shuffle data
    indices = [ x for x in range(batch_size) ]
    np.random.shuffle(indices)

    shuffled_inputs = np.array( [ np.array(inputs[idx]) for idx in indices ] )
    shuffled_labels = np.array( [ np.array(labels[idx]) for idx in indices ] )

    # Save to files
    image_output = open("math.training.images", "wb")
    pickle.dump(inputs,image_output)
    
    label_output = open("math.training.labels", "wb")
    pickle.dump(labels, label_output)
create_data(47)
