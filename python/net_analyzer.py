import caffe
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import os

caffe.set_mode_cpu()

if len(sys.argv) != 4:
	exit("Error: Incorrect number of arguments. \nUsage: net_analyzer.py <file path to model prototxt> <file path to .caffemodel file> <filepath to testing data>")

if not os.path.isfile(sys.argv[1]):
	exit("Error: File path to net prototxt is invalid.")
if not os.path.isfile(sys.argv[2]):
	exit("Error: File path to .caffemodel file is invalid.")
if not os.path.isfile(sys.argv[3]):
	exit("Error: File path to testing data is invalid.")

data = h5py.File(sys.argv[3], 'r')



refNet = caffe.Net(str(sys.argv[1]), str(sys.argv[2]), caffe.TEST)
net = caffe.Net(str(sys.argv[1]), str(sys.argv[2]), caffe.TEST)

reference = np.zeros((1, refNet.blobs['data'].data.shape[1]))
refNet.blobs['data'].data[...] = reference
#double check this starting point
#refNet.forward(start='Inner1')
refNet.forward(start='inner1')

for layer in net.layers:
	print layer.type
for name in net._layer_names:
	print name

finalValues = [1] * data.get('data').shape[1]
for k in range(data.get('data').shape[0]):
	multipliers = tuple()
	layerTypes = []
	for i in range(net.layers.__len__()):
		layerType = net.layers[i].type
		if layerType == "InnerProduct":
			multipliers += tuple(net.layers[i].blobs[0].data)
			layerTypes.append(layerType)
		elif layerType == "TanH" or layerType == "Sigmoid" or layerType == "ReLU":
			multipliers = tuple(net.blobs[net._layer_names[i]].data - refNet.blobs[net._layer_names[i]].data)
			layerTypes.append("Nonlinearity")
		else:
			continue
	currentValues = []
	#populate currentValues
	for value in multipliers[multipliers.shape[0] - 1]:
		currentValues.append(value[0])
	for i in range(layerTypes.__len__()):
		#have to go backwards in multiplier tuple
		pos = multipliers.shape[0] - i
		newVals = []
		for j in range(multipliers[pos].shape[1]):
			if layerTypes[pos] == "InnerProduct":
				temp = 0
				for l in range(multipliers[pos].shape[0]):
					temp += multipliers[pos][l][j] * currentValues[l]
				newVals.append(temp)
			else:
				newVals.append(currentValues[j] * multipliers[pos][0][j])
		currentValues = newValues

	deltaInput = net.blobs['data'].data - refNet.blobs['data'].data
	for i in range(deltaInput.shape[1]):
		currentValues[i] *= deltaInput[0][i]
		finalValues[i] += currentValues[i]
	
plt.bar(range(currentValues.__len__()), currentValues)
plt.savefig('images/importance.png')
plt.close()
print "Analysis Complete"
		
