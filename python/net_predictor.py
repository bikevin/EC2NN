import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import os.path
import numpy as np
from pylab import *

#use CPU instead of non-existent GPU
caffe.set_mode_cpu()

#check for correct number of input args, fail with error statement
if len(sys.argv) != 4:
        exit("Error: Incorrect number of arguments. \nUsage: net_predictor.py <path to model file> <path to caffemodel file> <num points to predict> <userdir>")

#check for valid userdir
if not os.path.exists(sys.argv[4]):
        exit("Error: Invalid userdir")

#append userdir to model/caffemodel file paths
netFilePath = sys.argv[4] + '/' + sys.argv[1]
modelFilePath = sys.argv[4] + '/' + sys.argv[2]

#check if the file paths are correct
if not os.path.isfile(netFilePath):
        exit("Error: File path to net prototxt is invalid.")
if not os.path.isfile(modelFilePath):
        exit("Error: File path to .caffemodel file is invalid.")
        
#make sure num points is actually an int
try:
        int(sys.argv[3])
except ValueError:
        exit("Error: Invalid number of points")
        
predictOutput = []

#initialize the network from pretrained model
net = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)

#for each point, compute the output and append that to the array
for i in range(int(sys.argv[3])):
        net.forward()
        predictOutput.append(cpy.deepcopy(net.blobs[net._blob_names[-1]].data))

#make sure the predict network doesn't have a loss layer, but there's no real check for that

#save the output to file "predict.out"
np.savetxt(sys.argv[4] + '/predict.out', predictOutput, delimiter=',')

