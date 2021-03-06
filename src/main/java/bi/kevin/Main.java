package bi.kevin;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collection;

public class Main {


    public static void main(String[] args) {

        //FORMAT OF A NEURAL NETWORK:
        //---------------------------------------------------------------
        //At least one data layer - set neuron count to 0, this is ignored,
        //but batchSize and filepath required, layerType must be set to 4
        //to be recognized as a data layer
        //
        //At least one computational layer - layerType, numNeurons required
        //
        //At least one output layer - layerType, numNeurons required
        //
        //Regression/Classification for output layer is defined by the
        //number of neurons in the output layer - 1 neuron is regression,
        //more than that will be classification.
        //
        //Phases is optional for every layer, and is generally only useful
        //when defining separate train/test data layers so one can use
        //different data files and batch sizes.

        //CREATE AN EXAMPLE NEURAL NETWORK

        //create array of layers to pass into createNet
        Layer[] layers = new Layer[4];

        //training data layer - layer type 4 (HDF5Data), neurons ignored,
        //data location is "training_files", training phase, batch size of 100
        layers[0] = new Layer(4, 0, "training_files", 1, 100);

        //testing data layer - layer type 4 (HDF5Data), neurons ignored,
        //data location is "test_files", testing phase, batch size of 100
        layers[1] = new Layer(4, 0, "testing_files", 2, 100);

        //hidden layer - layer type 2 (TanH), 35 neurons
        layers[2] = new Layer(2, 35);

        //output layer - layer type 2 (TanH), 1 neuron (for regression, more
        //for classification)
        layers[3] = new Layer(2, 2);

        //create a NetGenerator object - only one constructor, takes array of layers
        NetGenerator netGenerator = new NetGenerator(layers, "test");
        System.out.println("Example Neural Network:");
        System.out.println(netGenerator.createNet(false));
        System.out.println(netGenerator.createNet(true));

        //FORMAT OF A SOLVER:
        //---------------------------------------------------------------
        //Requires filepaths to neural net models, such as those outputted
        //from NetGenerator.createNet();
        //
        //Test iterations, test interval, snapshot frequency, gamma, power,
        //momentum, weight decay rate, base learning rate are all optional,
        //they have defaults - but it is highly recommended that you set
        //them yourself.
        //
        //It is probably also a good idea to iterate over these variables
        //in some kind of a grid search.
        //
        //VARIABLE EXPLANATIONS:
        //Test iterations is the number of iterations run when testing
        //
        //Test interval is the number of training iterations run before testing
        //
        //Snapshot frequency is how often the solver saves a snapshot of the
        //current training state
        //
        //The actual learning rate is calculated using this function:
        // actual_lr = base_lr * (1 + gamma * iter) ^ (- power)
        //
        //Higher weight decay means model is more general, but likely fits worse.
        //
        //Momentum increases the step size towards the minimum - helps
        //when you're getting stuck in local minima, but may also step
        //right over the global minimum.

        //Constructors: only the filepaths, or everything.

        //CREATE EXAMPLE SOLVER

        //solver with defaults
        SolverGenerator solverGenerator = new SolverGenerator("train_net.prototxt", "test_net.prototxt");
        System.out.println("Example Solver with Defaults");
        System.out.println(solverGenerator.createSolver());

        //solver with custom values
        solverGenerator = new SolverGenerator("train_net.prototxt", "test_net.prototxt", 50, (float) 0.1,
                (float) 0.5, (float) 0.5, (float) 0, (float) 0.00001, "test");
        System.out.println("Example Solver with Customization");
        System.out.println(solverGenerator.createSolver());

        //EXAMPLE STUFF - TO USE, UNCOMMENT AND REPLACE FILE PATHS WITH YOUR OWN
        //sends a file to server and deletes it
//
//        EC2Comm ec2Comm = new EC2Comm("ec2-54-152-208-18.compute-1.amazonaws.com", "/home/kevin/Downloads/neuralnetwork.pem", "test");
//        ec2Comm.trainNet("solver", 100, 50, 3, new String[0]);
/*        String[] filePaths = {"/home/kevin/Documents/NN/beets/beet_net.prototxt",
                "/home/kevin/Documents/NN/beets/beet_solver.prototxt",
                "/home/kevin/Documents/NN/beets/training_files",
                "/home/kevin/Documents/NN/beets/testing_files",
                "/home/kevin/Documents/NN/beets/train.h5",
                "/home/kevin/Documents/NN/beets/validate.h5"};*/

 //       ec2Comm.transferFilesToServer(filePaths, "");
 //       ec2Comm.trainNet("beet_solver.prototxt", 100, 50, 3, new String[0]);
//        ec2Comm.drawNet("beet_net.prototxt", "image.png");
 //       ec2Comm.transferOutputsToLocal("/home/kevin/Documents");
//        ec2Comm.cleanUp();


        try{

            Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

            Collection<? extends Collection<Double> > testData = Test.getResult();

            int[] labelIndicies = {3, 4};

            DataFormatter dataFormatter = new DataFormatter(labelIndicies, testData);

            LocalNetwork localNetwork = new LocalNetwork(dataFormatter.getAllDataNormalized(), 0.65, 100, layers, true);
            ArrayList<ModelInfo> netOutput = localNetwork.trainModel(new CustomListener(
                    localNetwork.getTrainData(),
                    localNetwork.getTestData(),
                    10,
                    new ArrayList<>()));

            for(ModelInfo info : netOutput){
                System.out.println(info.getIteration());
                System.out.println(info.getTestScore());
                System.out.println(info.getTrainScore());
            }

            Evaluation eval = localNetwork.getEval();

            System.out.println(eval.accuracy());

            MultiLayerNetwork network = localNetwork.getTrainedModel();

            String json = localNetwork.toJson();

            LocalNetwork newNet = new LocalNetwork(dataFormatter.getAllDataNormalized(), 0.65, json);
            MultiLayerNetwork newNetwork = newNet.getTrainedModel();

            INDArray params = network.getLayer(1).getParam("W");
            String newJson = newNet.toJson();

            ArrayList<INDArray> importances = localNetwork.getTrainImportance();

            importances = localNetwork.getTestImportance();

        } catch(Exception e){
            e.printStackTrace();
        }


    }


}


class Test {
    private static final String pathToExpandedColumns = "C:\\Users\\Kevin\\Downloads\\expanded.txt";

    public static Collection< ? extends Collection< Double > > getResult() throws Exception{

        Collection< Collection< Double > > result = new ArrayList();

        try( BufferedReader br = new BufferedReader( new FileReader( new File( pathToExpandedColumns ) ) ) ){
            String line;

            //iterate through lines in the input file
            while( (line=br.readLine()) != null ){

                //skip empty lines and lines that don't start with tabs
                if( line.trim().isEmpty() || !line.startsWith( "\t" ) )
                    continue;

                //parse floats
                Collection< Double > resultRow = new ArrayList();
                String[] parts = line.trim().split( "\t" );
                for( String s : parts )
                    resultRow.add( Double.parseDouble( s ) );
                result.add( resultRow );
            }
        }

        return result;
    }
}





