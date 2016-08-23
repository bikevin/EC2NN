import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import os.path
import numpy as np
from pylab import *

caffe.set_mode_cpu()

if len(sys.argv) != 5:
        exit("Error: Incorrect number of arguments. \nUsage: net_predictor.py <$")

if not os.path.exists(sys.argv[4]):
        exit("Error: Invalid userdir")

netFilePath = sys.argv[4] + '/' + sys.argv[1]
modelFilePath = sys.argv[4] + '/' + sys.argv[2]

if not os.path.isfile(netFilePath):
        exit("Error: File path to net prototxt is invalid.")
if not os.path.isfile(modelFilePath):
        exit("Error: File path to .caffemodel file is invalid.")

net = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)
net.forward()

#make sure the predict network doesn't have a loss layer, but there's no real c$

np.savetxt(sys.argv[4] + '/predict.out', [net.blobs[net._blob_names[-1]].data], delimiter=',')

