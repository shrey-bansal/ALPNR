import numpy as np
import random
# Training image file name
training_image_fn = "mnist/train-images.idx3-ubyte"

#Training label file name
training_label_fn = "mnist/train-labels.idx1-ubyte"

#Weights file name
model_fn = "model-neural-network.dat"

#Report file name
report_fn = "training-report.dat"

#Number of training samples
nTraining = 60000

#Image size in MNIST database
width = 28
height = 28

# n1 = Number of input neurons
# n2 = Number of hidden neurons
# n3 = Number of output neurons
# epochs = Number of iterations for back-propagation algorithm
# learning_rate = Learing rate
# momentum = Momentum (heuristics to optimize back-propagation algorithm)
# epsilon = Epsilon, no more iterations if the learning error is smaller than epsilon

n1 = width * height # = 784, without bias neuron
n2 = 128
n3 = 10; # Ten classes: 0 - 9
epochs = 512
learning_rate = 1e-3
momentum = 0.9
epsilon = 1e-3

# From layer 1 to layer 2. Or: Input layer - Hidden layer
w1,out1,delta1 = 0,0,0

# From layer 2 to layer 3. Or; Hidden layer - Output layer
w2, delta2, in2, out2, theta2 = 0,0,0,0,0

# Layer 3 - Output layer
in3, out3, theta3 = 0,0,0
expected = np.zeros(n3 + 1)

# Image. In MNIST: 28x28 gray scale images.
d = np.zeros((width+1,height+1))

# File stream to read data (image, label) and write down a report
image = 0
label = 0



def about():
    print("**************************************************")
    print("*** Training Neural Network for MNIST database ***")
    print("**************************************************")
    print()
    print("No. input neurons: " + str(n1))
    print( "No. hidden neurons: " + str(n2) )
    print( "No. output neurons: " + str(n3) )
    print()
    print("No. iterations: " + str(epochs) )
    print( "Learning rate: " + str(learning_rate) )
    print( "Momentum: " + str(momentum) )
    print( "Epsilon: " + str(epsilon) )
    print()
    print("Training image data: " + str(training_image_fn))
    print( "Training label data: " + str(training_label_fn))
    print( "No. training sample: " + str(nTraining))



def init_array():
	# Layer 1 - Layer 2 = Input layer - Hidden layer
    w1 = np.zeros((n1+1,n2+1))
    delta1 = np.zeros((n1+1,n2+1))
    out1 = np.zeros(n1+1)

	# Layer 2 - Layer 3 = Hidden layer - Output layer
    w2 = np.zeros((n2+1,n3+1))
    delta2 = np.zeros((n2+1,n3+1))
    out2 = np.zeros(n2+1)
    in2 = np.zeros(n2 + 1)
    theta2 = np.zeros(n2+1)

	# Layer 3 - Output layer
    in3 = np.zeros(n3 + 1)
    out3 = np.zeros(n3+1)
    theta3 = np.zeros(n3 + 1)

    # Initialization for weights from Input layer to Hidden layer
    for i in range(1,n1+1):
        for j in range(1,n2+1):
            number = random.randint(0,10000000)
            sign = number % 2;

            # Another strategy to randomize the weights - quite good
            # w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);

            w1[i][j] = (number % 6) / 10.0
            if (sign == 1):
				w1[i][j] = - w1[i][j]


	# Initialization for weights from Hidden layer to Output layer
    for i in range(1,n2+1):
        for j in range(1,n3+1):
            number = random.randint(0,10000000)
            sign = number % 2;

            # Another strategy to randomize the weights - quite good
            # w1[i][j] = (double)(rand() % 10 + 1) / (10 * n2);

            w2[i][j] = (number % 6) / 10.0
            if (sign == 1):
				w2[i][j] = - w2[i][j]


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

# +------------------------------+
# | Forward process - Perceptron |
# +------------------------------+

def perceptron():
    for i in range(1,n2+1):
		in2[i] = 0.0

    for i in range(1,n3+1):
		in3[i] = 0.0

    for i in range(1,n1+1):
        for j in range(1,n2+1):
            in2[j] += out1[i] * w1[i][j]

    for i in range(1,n2+1):
		out2[i] = sigmoid(in2[i])

    for i in range(1,n2+1):
        for j in range(1,n3+1):
            in3[j] += out2[i] * w2[i][j]

    for i in range(1,n3+1):
		out3[i] = sigmoid(in3[i])


# +---------------+
# | Norm L2 error |
# +---------------+

def square_error():
    res = 0.0
    for i in range(1,n3+1):
        res += (out3[i] - expected[i]) * (out3[i] - expected[i])
    res *= 0.5
    return res


# +----------------------------+
# | Back Propagation Algorithm |
# +----------------------------+

def back_propagation():
    sum

    for i in range(1,n3+1):
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i])

    for i in range (1,n2+1):
        sum = 0.0;
        for j in range(1,n3+1):
            sum += w2[i][j] * theta3[j]

        theta2[i] = out2[i] * (1 - out2[i]) * sum

    for i in range(1,n2+1):
        for j in range(1,n3+1):
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j])
            w2[i][j] += delta2[i][j]

    for i in range(1,n1+1):
        for j in range(1,n2+1):
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j])
            w1[i][j] += delta1[i][j]

# +-------------------------------------------------+
# | Learning process: Perceptron - Back propagation |
# +-------------------------------------------------+

def learning_process():
    for i in range(1,n1+1):
        for j in range(1,n2+1):
			delta1[i][j] = 0.0


    for i in range(1,n2+1):
        for j in range(1,n3+1):
			delta2[i][j] = 0.0

    for i in range(1,epochs+1):
        perceptron()
        back_propagation()
        if (square_error() < epsilon):
			return i
    return epochs


# +--------------------------------------------------------------+
# | Reading input - gray scale image and the corresponding label |
# +--------------------------------------------------------------+

def input():
	# Reading image

    for j in range(1,height+1):
        for i in range(1,width+1):
            number = image.read(1)
            if(number==0):
			    d[i][j] = 0
            else:
				d[i][j] = 1

    for j in range(1,height+1):
        for i in range(1,width+1):
            pos = int(i + (j - 1) * width)
            out1[pos] = d[i][j]

	# Reading label
    number = label.read(1);
    for i in range(1,n3+1):
		expected[i] = 0.0
    expected[number + 1] = 1.0

    print("Label: " +(int)(number))


# +------------------------+
# | Saving weights to file |
# +------------------------+

def write_matrix(file_name):
    file = open(file_name,"r+")
	# Input layer - Hidden layer
    for i in range(1,n1+1):
        for j in range(1,n2+1):
			file.write(w1[i][j]+ " ")
        str = ""
        L = [str]
        file.writelines(L)

	# Hidden layer - Output layer
    for i in range(1,n2+1):
        for j in range(1,n3+1):
		    file.write(w2[i][j]+ " ")
        file.writelines(L)
    file.close()

# +--------------+
# | Main Program |
# +--------------+

about()

report = open(report_fn,"r+")
image = open(training_image_fn,"r+")
label = open(training_label_fn,"r+")



for i in range(1,17):
    number = image.read(1)

for i in range(1,9):
    number = label.read(1)

#Neural Network Initialization
init_array()
	#time_req = clock();
for sample in range(1,nTraining//500+1):
    print( "Sample " ,sample )

    # Getting (image, label)
    input()

	# Learning process: Perceptron (Forward procedure) - Back propagation
    nIterations = learning_process()

	# Write down the squared error
    print( "No. iterations: " + nIterations )
    print("Error: %0.6lf\n\n", square_error())
    report.write( "Sample " +sample +": No. iterations = " +nIterations + ", Error = " + square_error() )

	# Save the current network (weights)
    if (sample % 100 == 0) :
		print( "Saving the network to " +model_fn + " file." )
		write_matrix(model_fn)


	#time_req = clock() - time_req;
	#cout<<"Time required: "<<(float)time_req/CLOCKS_PER_SEC << " seconds"<<)
# Save the final network
write_matrix(model_fn)

report.close()
image.close()
label.close()
