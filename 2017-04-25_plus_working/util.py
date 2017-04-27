
# Read each equation and its label from training/test set
def print_equation(inputs, labels):
#    print "4 = 4"
#    print "4 + 2 = 1"
   
    equation_label = ""
    # Count to see how many element in vector equal 1
    # If count == 1 that means the equation only have 1 operand (without operator)
    # so the printed equation will be in format "__1 = 1" for the equations have only the first operand
    #or "1__ = 1" for the equations have only the second operand
    count = 0
    # Get the equation string from vectors of input and label
    equation = ""
    for i in range(len(inputs)):
        if inputs[i]==1:
            # Start counting everytime the element is equal 1, not 0
            count += 1
            # Get the updated index of the element having value of 1
            # to see if it is the first operand (index < 10)
            # or it is the second operand (index > 11)
            index = i
            if i<10:
                equation += str(i)
            elif i == 10:
                equation += '+'
            elif i == 11: 
                equation += '-'
            else:
                equation += str(i%12)
    # If count == 1 => 1-operand equation printed in format "__1" or "__1"
    if count == 1:
        if index < 10:
            equation_label = equation +  "__"
        else:
            equation_label = "__" + equation
    # if count != 1 => normal equations with 2 operand and a operator
    else:
        equation_label += equation
    # Get the equal sign for equation
    equation_label += " = "
    # Get the result of equation
    #for i in range (len(labels)):
        #if labels[i] == 1:
            #equation_label += str(i)
    equation_label += str(int(labels[0]))
    print equation_label
              
