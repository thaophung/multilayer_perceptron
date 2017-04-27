import numpy as np
import random
import pickle
from util import print_equation

# create input_vector and label_vector for each equation
def create_vector(equation):
    input_vector = np.zeros((22))
    label_vector = np.zeros((1))

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
    label_vector[0] =int(result)

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

    print test_labels
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
                equ_label = np.zeros((1))
                equ_vector[j] = 1
                equ_label[0] = int(j)
                # Add equations and labels to training_inputs and training_labels  arrays
                training_inputs.append(equ_vector)
                training_labels.append(equ_label)
        
        # Gnerate equations have only second operand
        elif (equ_group == 'third'):
            for j in range(0,10):
                equ_vector = np.zeros((22))
                equ_label = np.zeros((1))
                equ_vector[12+j] = 1
                equ_label[0] = int(j)
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
    print len(training_inputs) 
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
    heldout=['0+0','0+1','1+0','0+2', '2+0','0+3','3+0','0+4','4+0', '0+5','5+0', '0+6','6+0','0+7', '7+0', '0+8', '8+0',
            '1+1','1+2', '2+1', '1+3','3+1','1+4','4+1','1+5','5+1','1+6','6+1','1+7','7+1','1+8','8+1','1+9','9+1'
            '2+2', '2+3', '3+2','2+4', '4+2','2+5','5+2','2+6','6+2','2+7','7+2','2+8','8+2','2+9','9+2',
            '3+3','3+4','4+3','3+5','5+3','3+6','6+3','3+7','7+3','3+8','8+3','3+9','9+3'
            '4+4','4+5','5+4','4+6','6+4','4+7','7+4','4+8','8+4','4+9','9+4'
            '5+5','5+6','6+5','5+7','7+5','8+5','5+8','5+9','9+5'
            '6+6','6+7','7+6','6+8','8+6','6+9','9+6'
            '7+7','7+8','8+7','7+9','9+7',
            '8+8','8+9','9+8'
            '9+9']

    # Write the equations you want to generate: eg. +1, 1+, 0+, +0,
    # For equations have only 1 digit, 
    #       first operand: put 'first'
    #       second operand: put 'third'
    equations = ['first', 'third', '0+', '+0']

    create_data(heldout, equations)

if __name__ == "__main__":
    main()
