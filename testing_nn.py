import numpy as np
testing_image_fn = "mnist/t10k-images.idx3-ubyte"

testing_label_fn = "mnist/t10k-labels.idx1-ubyte"

model_fn = "model-neural-network.dat"

report_fn = "testing-report.dat"

nTesting = 10000

 # Image size in MNIST database
width = 28
height = 28

 # n1 = Number of input neurons
 # n2 = Number of hidden neurons
 # n3 = Number of output neurons

n1 = width * height   # = 784, without bias neuron
n2 = 128
n3 = 10   # Ten classes: 0 - 9

 # From layer 1 to layer 2. Or: Input layer - Hidden layer


 # From layer 2 to layer 3. Or  Hidden layer - Output layer


 # Layer 3 - Output layer

expected = np.zeros(n3+1)

 # Image. In MNIST: 28x28 gray scale images.
d = np.zeros((width+1,height+1))


 # +--------------------+
 # | About the software |
 # +--------------------+

def about():
	 # Details
	print( "*************************************************" )
	print( "*** Testing Neural Network for MNIST database ***" )
	print( "*************************************************" )
	print( )
	print( "No. input neurons: " ,n1 )
	print( "No. hidden neurons: " ,n2 )
	print( "No. output neurons: ", n3 )
	print( )
	print( "Testing image data: " ,testing_image_fn )
	print( "Testing label data: " , testing_label_fn )
	print( "No. testing sample: " , nTesting )


 # +-----------------------------------+
 # | Memory allocation for the network |
 # +-----------------------------------+


 # Layer 1 - Layer 2 = Input layer - Hidden layer
w1 = np.zeros((n1+1,n2+1))


out1 = np.zeros(n1 + 1 )

 # Layer 2 - Layer 3 = Hidden layer - Output layer
w2 = np.zeros((n2+1,n3+1))

in2 = np.zeros(n2 + 1)
out2 = np.zeros(n2 + 1)

 # Layer 3 - Output layer
in3 = np.zeros(n3 + 1)
out3 = np.zeros(n3 + 1)


 # +----------------------------------------+
 # | Load model of a trained Neural Network |
 # +----------------------------------------+

def load_model(file_name):
	file = open(file_name, "r+")
	for i in range(1,n1+1):
		for j in range(1,n2+1):
		    file.write(str(w1[i][j]))
		file.writelines([""])
	for i in range(1,n2+1):
		for j in range(1,n3+1):
			file.write(str(w2[i][j]))
		file.writelines([""])


	file.close()


 # +------------------+
 # | Sigmoid function |
 # +------------------+

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

 # +--------------------------------------------------------------+
 # | Reading input - gray scale image and the corresponding label |
 # +--------------------------------------------------------------+
number =0
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
    number = str(label.read(1))
    for i in range(1,n3+1):
		expected[i] = 0.0
    expected[ int(number)+1] = 1.0

    print("Label: " , int(number))


 # | Main Program |
 # +--------------+


about()

report = open(report_fn,"r+")
image = open(testing_image_fn,"r+")
label = open(testing_label_fn,"r+")

 # Reading file headers
for i in range(1,17):
    number = image.read(1)

for i in range(1,9):
    number = label.read(1)

 # Neural Network Initialization
#init_array()   # Memory allocation
load_model(model_fn)   # Load model (weight matrices) of a trained Neural Network

nCorrect = 0
#	clock_t time_req
#	time_req = clock()
for sample in range(1,nTesting+1):
    print( "Sample ",sample )

     # Getting (image, label)
    label = input()

	 # Classification - Perceptron procedure
    perceptron()

     # Prediction
    predict = 1
    for i in range(2,n3+1):
		if (out3[i] > out3[predict]):
			predict = i


    predict-=1

	 # Write down the classification result and the squared error
    error = square_error()
    print("Error: %0.6lf\n", error)
    if (label == predict):
		nCorrect+=1
		print( "Classification: YES. Label = " ,label , ". Predict = " , predict )
		report.write( "Sample " + str(sample) + ": YES. Label = " + str(label) + ". Predict = " + str(predict) + ". Error = " + str(error) )
    else:
		print( "Classification: NO.  Label = " , label , ". Predict = " , predict )
		print( "Image:" )
		for j in range(1,height+1):
			for i in range(1,width+1):
				print( d[i][j])
			print()

		print()
		report.write( "Sample " + str(sample) + ": NO.  Label = " + str(label) + ". Predict = " + str(predict) + ". Error = " + str(error) )


#	time_req = clock() - time_req
 # Summary
accuracy = (nCorrect) / nTesting * 100.0
print( "Number of correct samples: " , nCorrect , " / " , nTesting )
print("Accuracy: %0.2lf\n", accuracy)
	# cout<<"Time taken: "<<(float)time_req/CLOCKS_PER_SEC<<" seconds"<<endl
report.write( "Number of correct samples: " + str(nCorrect) + " / " + str(nTesting) )
report.write("Accuracy: " + str(accuracy) )
str()
report.close()
image.close()
label.close()
