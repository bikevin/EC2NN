import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import caffe
import h5py
import copy
import sys

caffe.set_mode_cpu()

#check arguments
if len(sys.argv) != 3:
	exit("Error: Incorrect number of arguments. \nUsage: net_analyzer.py <file path to model prototxt> <file path to .caffemodel file>")

#function that returns the correct function given a type
def derivPicker(neuronType):
        #define the partial derivative functions
        def SigDeriv(x):
                return x * (1 - x)

        def TanHDeriv(x):
                return (1 - x * x)

        def ReLUDeriv(x):
                if x > 0:
                        return 1
                else:
                        return 0
        if neuronType == "TanH":
                return TanHDeriv
        elif neuronType == "Sigmoid":
                return SigDeiv
        elif neuronType == "ReLU":
                return ReLUDeriv
        else:
                return 0

def calcInputDerivs(net, layerArray, index, currentValue, inputIndex, neuronIndex):
        layerType = layerArray[index]
        numNeuronsNext = net.layers[index + 1].blobs[0].data[inputIndex].shape(0)
        if :#how to get name of layer? check layer name here for base case - if it's SigmoidBottom, ReLUBottom, or TanHBottom
                for i in range(numNeuronsNext):
                        calcInputDerivativesBase(net, layerArray, index, currentValue, i);
        if layerType == "HDF5Data":
                return 0
        elif layerType == "InnerProduct":
                retVal = 0
                for i in range(numNeuronsNext):
                        retVal += calcInputDerivs(net, layerArray, index += 1, currentValue * net.layers[index].blobs[0].data[inputIndex][neuronIndex], inputIndex, i)
                return retVal
        else:
                retVal = 0
                for i in range(numNeuronsNext):
                        retVal += calcInputDerivs(net, layerArray, index += 1, currentValue * derivPicker(layerType)(net.layers[index].blobs[0].data[neuronIndex]), inputIndex, i)
                return retVal

def calcInputDerivsBase(net, layerArray, index, currentValue, neuronIndex):
        return currentValue * derivPicker(layerArray[index])(net.layers[index].blobs[0].data[neuronIndex])




#generate the net
net = caffe.Net(str(sys.argv[1]), str(sys.argv[2]), caffe.TEST)

for i in range(net.blob("data").shape(0)):
        for j in range(net.blob("inner1").shape(0)):
                print calcInputDerivs(net, net.layers.type, 0, i, j);


#TODO NEXT:
#holy shit test this pile of cancer

