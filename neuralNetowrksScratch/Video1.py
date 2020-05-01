# Fully conected Perceptron 4-3-3-1

# 4 neurons input layer
inputs = [0.1, 0.2, 0.9, 0.6]

# 3 neurons hidden layer
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

hiddenLayer1 = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1, 
		        inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
		  		inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]


# 3 neurons hidden layer
weights4 = [0.2, 0.8, -0.5, 1.0]
weights5 = [0.5, -1, 0.3, -0.5]
weights6 = [-0.26, -1, 0.17, 0.5]

bias4 = 2
bias5 = 3
bias6 = 0.5


hiddenLayer2 = [hiddenLayer1[0]*weights4[0] + hiddenLayer1[1]*weights4[1] + hiddenLayer1[2]*weights4[2] + bias4, 
		        hiddenLayer1[0]*weights5[0] + hiddenLayer1[1]*weights5[1] + hiddenLayer1[2]*weights5[2] + bias5,
		        hiddenLayer1[0]*weights6[0] + hiddenLayer1[1]*weights6[1] + hiddenLayer1[2]*weights6[2] + bias6]

# 1 neuron output layer
weights = [0.2, 0.8, 0.3, 0.5]
bias = 1

output = hiddenLayer2[0]*weights[0] + hiddenLayer2[1]*weights[1] + hiddenLayer2[2]*weights[2] + bias

print(inputs)
print(hiddenLayer1)
print(hiddenLayer2)
print(output)
