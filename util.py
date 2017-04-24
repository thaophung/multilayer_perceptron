

def print_equation(inputs, labels):
#    print "4 = 4"
#    print "4 + 2 = 1"
    equation_label = ""
    for i in range(len(inputs)):
        equation = ""
        if inputs[i]==1:
            if i<10:
                equation += str(i)
            elif i == 10:
                equation += '+'
            elif i == 11: 
                equation += '-'
            else:
                equation += str(i%12)
        equation_label += equation
    equation_label += " = "
    for i in range (len(labels)):
        if labels[i] == 1:
            equation_label += str(i)
    print equation_label
              
