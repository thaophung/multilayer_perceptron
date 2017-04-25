import numpy as np
import random
import pickle
from util import print_equation

# create input_vector and label_vector for each equation
def create_vector(equation):
    input_vector = np.zeros((22))
    label_vector = np.zeros((19))

    # Get the first operand
    operand1 = equation[0]
    int_operand1 = int(operand1)
    input_vector[int_operand1] = 1
    
    # Get the second operand
    operand2 = equation[2]
    int_operand2 = int(operand2)
    input_vector[12+int_operand2] = 1

    # Get the operator
    operator = equation[1]
    if operator == '+':
        input_vector[10] = 1
        result = int_operand1 + int_operand2
    else:
        input_vecotr[11] = 1
        result = int_operand1 - int_operand2
    label_vector[result] =1

    return input_vector, label_vector

# Create whole training and test set
def create_data(heldout, equations):

    # arrays for training and testing's inputs and labels 
    training_inputs=[]
    training_labels=[]
    test_inputs=[]
    test_labels=[]

    # Generate test set (heldout equations)
    for i in range(len(heldout)):
        equation = heldout[i]
        input_vector, label_vector = create_vector(equation)
        test_inputs.append(input_vector)
        test_labels.append(label_vector)

    print "Test set: "
    for i in range(len(test_inputs)):
        print_equation(test_inputs[i], test_labels[i])

    # Generate training set
    for i in range(len(equations)):
        equ_group = equations[i]
        
        # Generate equations have only first operand
        if (equ_group == 'first'):
            for j in range(0,10):
                equ_vector = np.zeros((22))
                equ_label = np.zeros((19))
                equ_vector[j] = 1
                equ_label[j] = 1
                # Add equations and labels to training_inputs and training_labels  arrays
                training_inputs.append(equ_vector)
                training_labels.append(equ_label)
        
        # Gnerate equations have only second operand
        elif (equ_group == 'third'):
            for j in range(0,10):
                equ_vector = np.zeros((22))
                equ_label = np.zeros((19))
                equ_vector[12+j] = 1
                equ_label[j] = 1
                #Add equations and labels to arrays
                training_inputs.append(equ_vector)
                training_labels.append(equ_label)

        # Gerate equations have 2 operands and 1 operator
        else:
            # if the equations have fixed first operand (eg. 1+, 2+)
            if (equ_group[0] != '+'):
                for j in range (0, 10):
                   
                    equation = equ_group + str(j)
                    # Test if the equation is already in heldout set
                    if equation in heldout:
                        continue

                    # if equation is not in heldout, get input and label vectors
                    input_vector, label_vector = create_vector(equation)
                    # Save to training_inputs and training_labels arrays
                    training_inputs.append(input_vector)
                    training_labels.append(label_vector)

            # If the equations have fixed second operand(eg. +1, +2)
            else:
                for j in range(0, 10):
                    equation = str(j) + equ_group
                    # Test if the equation is already in heldout set
                    if equation in heldout:
                        continue

                    # if equation is not in heldout, get input and label vectors
                    input_vector, label_vector = create_vector(equation)
                    # Save to training_inputs and training_labels arrays
                    training_inputs.append(input_vector)
                    training_labels.append(label_vector)

    # Print training set
    print("\nTraining set")
    for i in range(len(training_inputs)):
        print_equation(training_inputs[i], training_labels[i])
    
    # Save inputs and labels to files using pickle
    training_inputs_file = open("training_inputs.pkl", "wb")
    pickle.dump(training_inputs, training_inputs_file)

    training_labels_file = open("training_labels.pkl", "wb")
    pickle.dump(training_labels, training_labels_file)

    test_inputs_file = open("test_inputs.pkl", "wb")
    pickle.dump(test_inputs, test_inputs_file)

    test_labels_file = open("test_labels.pkl", "wb")
    pickle.dump(test_labels, test_labels_file)


def main():
    # Write the equations you want heldout in the helout list e.g 1+2, 4+3, 9-0
    heldout = ['1+2', '2+1', '1+6']

    # Write the equations you want to generate: eg. +1, 1+, 0+, +0,
    # For equations have only 1 digit, 
    #       first operand: put 'first'
    #       second operand: put 'third'
    equations = ['first', 'third', '0+', '+0', '1+', '+1', '+2']

    create_data(heldout, equations)

if __name__ == "__main__":
    main()
