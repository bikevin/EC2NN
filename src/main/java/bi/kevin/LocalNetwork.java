package bi.kevin;

import org.bytedeco.javacv.CanvasFrame;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by Kevin on 8/23/2016.
 */
public class LocalNetwork {

    private DataSet dataSet;
    private double testTrainSplit;
    private int iterations, numInputs, numOutputs;
    private Layer[] layerArray;
    private boolean isClassification;

    public LocalNetwork(DataSet dataSet, double testTrainSplit, int iterations, Layer[] layerArray, boolean isClassification){
        this.dataSet = dataSet;
        this.testTrainSplit = testTrainSplit;
        this.iterations = iterations;
        this.layerArray = layerArray;
        numInputs = dataSet.numInputs();
        numOutputs = dataSet.numOutcomes();
        this.isClassification = isClassification;
    }

    private String getActivation(Layer layer) throws Exception{
        int type = layer.getLayerType();

        if(type == 0){
            return "sigmoid";
        } else if (type == 1){
            return "tanh";
        } else if (type == 2){
            return "relu";
        } else {
            throw new Exception("Tried to get activation of a layer without an activation.");
        }
    }

    private MultiLayerConfiguration buildNet() throws Exception{
        NeuralNetConfiguration.ListBuilder conf = new NeuralNetConfiguration.Builder()
                .iterations(iterations)
                .weightInit(WeightInit.XAVIER)
                .regularization(true).l2(1e-4)
                .updater(Updater.ADADELTA)
                .momentum(0.9).rho(0.95)
                .list();

        int layerCount = 0;

        int prevSize = 0, nextSize = 0;

        for(int i = 1; i < layerArray.length - 1; i++){

            Layer prevLayer = layerArray[i - 1];
            Layer currentLayer = layerArray[i];
            Layer nextLayer = layerArray[i + 1];

            if(currentLayer.getPhase() != 2 && currentLayer.getLayerType() != 4){

                prevSize = prevLayer.getNeurons();
                nextSize = nextLayer.getNeurons();

                if(prevLayer.getLayerType() == 4){
                    prevSize = numInputs;
                }

                DenseLayer tempLayer = new DenseLayer.Builder().nIn(prevSize).nOut(nextSize)
                        .activation(getActivation(currentLayer)).build();

                conf.layer(layerCount++, tempLayer);

            }

        }

        Layer outputLayer = layerArray[layerArray.length - 1];

        if(isClassification){
            conf.layer(layerCount, new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                    .activation(getActivation(outputLayer))
                    .nIn(layerArray[layerArray.length - 2].getNeurons()).nOut(numOutputs).build());
        } else {
            conf.layer(layerCount, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(getActivation(outputLayer))
                    .nIn(layerArray[layerArray.length - 2].getNeurons()).nOut(numOutputs).build());
        }

        return conf.pretrain(false).backprop(true).build();
    }
}
