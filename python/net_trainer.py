import caffe
import sys
import os.path
import rsquared
from pylab import *

caffe.set_mode_cpu()

if len(sys.argv) != 5 and len(sys.argv) != 6 and len(sys.argv) != 7:
	exit("Error: Incorrect number of arguments. \nUsage: net_trainer.py <file path to solver> <iterations to train> <iterations between tests> <iterations per test> <model file path (optional)> <solverstate file path (optional)>")

if not os.path.isfile(sys.argv[1]):
	exit("Error: File path to solver is invalid.")
if len(sys.argv) >= 6:
	if not os.path.isfile(sys.argv[5]):
		exit("Error: File path to model file is invalid.")
	if len(sys.argv) == 7:
		if not os.path.isfile(sys.argv[6]):
			exit("Error: File path to solverstate file is invalid.")

try:
	int(sys.argv[2])
	int(sys.argv[3])
	int(sys.argv[4])
except ValueError:
	exit("Error: Iterations to train and iterations between tests must be integers")

if int(sys.argv[3]) == 0:
	exit("Error: Iterations between tests must be greater than zero")

solver = caffe.SGDSolver(sys.argv[1])

if len(sys.argv) == 6:
	solver.net.copy_from(sys.argv[5])
if len(sys.argv) == 7:
	solver.restore(sys.argv[6])

niter = int(sys.argv[2])
test_interval = int(sys.argv[3])

trainingNum = int(niter / test_interval)

train_loss = zeros(niter)
test_loss = zeros(trainingNum)
rsquared = zeros(trainingNum)

for it in range(niter):
	solver.step(1)

	train_loss[it] = solver.net.blobs['loss'].data
	
	if it % test_interval == 0:
		for test_it in range(int(sys.argv[4])):
			solver.test_nets[0].forward()	
		test_loss[int(it / test_interval)] = solver.test_nets[0].blobs['loss'].data
		rsquared[it / test_interval] = r2.calculateR2(solver.test_nets[0].blobs['innerBottom'].data, solver.test_nets[0].blobs['label'].data)

plot(arange(niter), train_loss)
savefig('train_loss.png')
close()
plot(arange(trainingNum), test_loss)
savefig('test_loss.png')
close()
plot(arange(trainingNum, rsquared))
savefig('rsquared.png')
close()

print "\nTraining complete."

