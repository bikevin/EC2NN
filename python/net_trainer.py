import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import os.path
import rsquared as r2
from pylab import *

#set to use CPU and not GPU
caffe.set_mode_cpu()

#check for enough arguments
if len(sys.argv) != 6 and len(sys.argv) != 7 and len(sys.argv) != 8 and len(sys.argv) != 9:
        exit("Error: Incorrect number of arguments. \nUsage: net_trainer.py <file path to solver> <iterations to train> <iterations between tests> <iterations per test> <userdir> <simple> <model file path (optional)> <solverstate file path (optional)>")

#initialize the paths
solverFilePath = sys.argv[5] + '/' + sys.argv[1]
modelFilePath = ''
stateFilePath = ''

#check for relevant paths
if not os.path.isfile(solverFilePath):
        exit("Error: File path to solver is invalid.")
if len(sys.argv) >= 8:
        if not os.path.isfile(modelFilePath):
                exit("Error: File path to model file is invalid.")
        modelFilePath = sys.argv[5] + '/' + sys.argv[7]
        if len(sys.argv) == 9:
                if not os.path.isfile(stateFilePath):
                        exit("Error: File path to solverstate file is invalid.")
                stateFilePath = sys.argv[5] + '/' + sys.argv[8]

#make sure ints are actually ints
try:
	int(sys.argv[2])
	int(sys.argv[3])
	int(sys.argv[4])
except ValueError:
	exit("Error: Iterations to train and iterations between tests must be integers")

#check for invalid inputs
if int(sys.argv[3]) == 0:
	exit("Error: Iterations between tests must be greater than zero")

#switch between simple solver (adadelta) and complex solver (stochastic gradient descent)
if sys.argv[6] == 'true':
	solver = caffe.AdaDeltaSolver(solverFilePath)
else:
	solver = caffe.SGDSolver(solverFilePath)

#allow starting from a set of weights or starting from pretrained gradients and weights
if len(sys.argv) == 8:
        solver.net.copy_from(modelFilePath)
        solver.test_nets[0].copy_from(modelFilePath)
if len(sys.argv) == 9:
        solver.restore(stateFilePath)
        solver.test_nets[0].restore(stateFilePath)

#checks are over, set values from the inputs
niter = int(sys.argv[2])
test_interval = int(sys.argv[3])

trainingNum = int(niter / test_interval)

train_loss = zeros(niter)
test_loss = zeros(trainingNum)
rsquared = zeros(trainingNum)

#manually step through the iterations so we can capture some other information
for it in range(niter):
	solver.step(1)

	#save the training loss for iteration
	train_loss[it] = solver.net.blobs['loss'].data
	
	#if it's testing time
	if it % test_interval == 0:
		
		#run through the test net for the correct number of iterations
		for test_it in range(int(sys.argv[4])):
			solver.test_nets[0].forward()	
		print solver.test_nets[0].blobs['loss'].data
		
		#save the test loss
		test_loss[int(it / test_interval)] = solver.test_nets[0].blobs['loss'].data
		
		#try saving the r^2
		try:
			rsquared[it / test_interval] = r2.calculateR2(solver.test_nets[0].blobs['innerBottom'].data, solver.test_nets[0].blobs['label'].data)
		#if it fails, that means it's a categorical and we use accuracy instead
		except ValueError:
			rsquared[it / test_interval] = solver.test_nets[0].blobs['accuracy'].data

#save everything
np.savetxt(sys.argv[5] + '/train_loss.out', train_loss, delimiter=',')
np.savetxt(sys.argv[5] + '/test_loss.out', test_loss, delimiter=',')
np.savetxt(sys.argv[5] + '/rsquared.out', rsquared, delimiter=',')

print "\nTraining complete."

