   

def print_equation(feature, label):
#    print "4 = 4"
#    print "4 + 2 = 1"
    equation_label = ""
    for i in range(len(feature)):
        equation = ""
        if feature[i]==1:
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
    for i in range (len(label)):
        if label[i] == 1:
            equation_label += str(i)
    print equation_label
              
