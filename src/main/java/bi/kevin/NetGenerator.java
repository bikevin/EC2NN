package bi.kevin;

import com.google.protobuf.TextFormat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kevin on 7/25/16.
 */
public class NetGenerator {

    private Layer[] netLayers = new Layer[0];
    private String userDir = "";

    public NetGenerator(Layer[] layers, String userDir){
        this.netLayers = layers;
        this.userDir = userDir;
    }

    //What do we care about for the network design?
    //Number of layers -> Layer object contains information about neuron type and layer size
    //See above for defining whether neural net computes classification or regression
    public String createNet(Boolean predict){

        Layer[] layers = netLayers;

        List<Caffe.LayerParameter> parameters = new ArrayList<>();
        //count the number of data layers
        int dataLayerCount = 0;
        for (Layer layer : layers) {
            if (layer.getLayerType() == 4) {
                dataLayerCount++;
            }
        }

        if(dataLayerCount == 0){
            System.err.println("Requires at least one data layer");
            return "ERROR";
        } else if(dataLayerCount > 2){
            System.err.println("Can't have more than two data layers");
            return "ERROR";
        } else if(dataLayerCount == 2){
            if(layers[0].getPhase() == 0 || layers[1].getPhase() == 0){
                System.err.println("Phases must be defined for two data layers");
                return "ERROR";
            }
        }

        //build the data layers
        for(int i = 0; i < dataLayerCount; i++) {
            parameters.add(dataLayerBuilder(layers[i]));
        }

        //build the hidden layers
        Caffe.LayerParameter.Builder[] currentLayer;
        for(int i = dataLayerCount; i < layers.length - 1; i++){
            currentLayer = hiddenLayerBuilder(layers[i], i - dataLayerCount + 1);
            currentLayer[0].addBottom(parameters.get(parameters.size() - 1).getTop(0));

            for(Caffe.LayerParameter.Builder builder : currentLayer){
                parameters.add(builder.build());
            }
        }

        //build the output layers
        currentLayer = outputLayerBuilder(layers[layers.length - 1], predict);
        currentLayer[0].addBottom(parameters.get(parameters.size() - 1).getTop(0));

        for(Caffe.LayerParameter.Builder builder : currentLayer){
            parameters.add(builder.build());
        }


        //finally, build the neural net
        Caffe.NetParameter.Builder neuralNet = Caffe.NetParameter.newBuilder();

        for(Caffe.LayerParameter parameter : parameters){
            neuralNet.addLayer(parameter);
        }

        //return a string representation
        return TextFormat.printToString(neuralNet);
    }

    private Caffe.LayerParameter.Builder[] hiddenLayerBuilder(Layer layer, int index){
        Caffe.LayerParameter.Builder[] outParams = new Caffe.LayerParameter.Builder[2];
        Caffe.LayerParameter.Builder hiddenLayer = Caffe.LayerParameter.newBuilder();
        String[] possibleNames = {"undef", "Sigmoid", "TanH", "ReLU", "HDF5Data"};

        hiddenLayer.setName("inner" + String.valueOf(index)).setType("InnerProduct")
                .setInnerProductParam(Caffe.InnerProductParameter.newBuilder().setNumOutput(layer.getNeurons())
                        .setWeightFiller(Caffe.FillerParameter.newBuilder().setType("xavier"))).addTop("inner" + String.valueOf(index));
        outParams[0] = hiddenLayer;

        hiddenLayer = Caffe.LayerParameter.newBuilder();
        hiddenLayer.setName(possibleNames[layer.getLayerType()] + String.valueOf(index))
                .setType(possibleNames[layer.getLayerType()]).addTop(possibleNames[layer.getLayerType()] + String.valueOf(index))
                .addBottom("inner" + String.valueOf(index));
        outParams[1] = hiddenLayer;

        if(layer.getPhase() == 1){
            outParams[0].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TRAIN));
            outParams[1].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TRAIN));
        } else if(layer.getPhase() == 2){
            outParams[0].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TEST));
            outParams[1].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TEST));
        }

        return outParams;
    }

    //input layer
    private Caffe.LayerParameter dataLayerBuilder(Layer layer){
        Caffe.LayerParameter.Builder dataLayer = Caffe.LayerParameter.newBuilder();
        dataLayer.setName("data").setType("HDF5Data").addTop("data").addTop("label")
                .setHdf5DataParam(Caffe.HDF5DataParameter.newBuilder()
                        .setSource(userDir + "/" + layer.getDataFile()).setBatchSize(layer.getBatchSize()).build());
        if(layer.getPhase() == 1){
            dataLayer.addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TRAIN));
        } else if(layer.getPhase() == 2){
            dataLayer.addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TEST));
        }

        return dataLayer.build();
    }

    //output classification layer- MUST HAVE A NEURON TYPE
    private Caffe.LayerParameter.Builder[] outputLayerBuilder(Layer layer, Boolean predict){
        Caffe.LayerParameter.Builder outputLayer = Caffe.LayerParameter.newBuilder();
        String[] possibleNames = {"undef", "Sigmoid", "TanH", "ReLU", "HDF5Data"};

        Caffe.LayerParameter.Builder[] outParams;

        if(predict){
            outParams = new Caffe.LayerParameter.Builder[3];
        } else {
            outParams = new Caffe.LayerParameter.Builder[4];
        }



        outputLayer.setName("innerBottom").setType("InnerProduct").setInnerProductParam(Caffe.InnerProductParameter.newBuilder()
                .setNumOutput(layer.getNeurons()).setWeightFiller(Caffe.FillerParameter.newBuilder().setType("xavier"))).addTop("innerBottom");
        outParams[0] = outputLayer;

        outputLayer = Caffe.LayerParameter.newBuilder();

        outputLayer.setName("NeuronBottom")
                .setType(possibleNames[layer.getLayerType()]).addTop("NeuronBottom")
                .addBottom("innerBottom");
        outParams[1] = outputLayer;


        outputLayer = Caffe.LayerParameter.newBuilder();

        outputLayer.setName("accuracy").setType("Accuracy").addBottom("NeuronBottom").addBottom("label")
                .addTop("accuracy");
        outParams[2] = outputLayer;


        outputLayer = Caffe.LayerParameter.newBuilder();

        if(!predict) {
            if (layer.getNeurons() == 1) {
                outputLayer.setName("loss").setType("EuclideanLoss").addBottom("NeuronBottom").addBottom("label")
                        .addTop("loss");
                outParams[3] = outputLayer;
            } else {
                outputLayer.setName("loss").setType("SoftmaxWithLoss").addBottom("NeuronBottom").addBottom("label")
                        .addTop("loss");
                outParams[3] = outputLayer;
            }
            if(layer.getPhase() == 1){
                outParams[3].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TRAIN));
            } else if(layer.getPhase() == 2){
                outParams[3].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TEST));
            }

        }

        if(layer.getPhase() == 1){
            outParams[0].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TRAIN));
            outParams[1].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TRAIN));
        } else if(layer.getPhase() == 2){
            outParams[0].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TEST));
            outParams[1].addInclude(Caffe.NetStateRule.newBuilder().setPhase(Caffe.Phase.TEST));
        }

        return outParams;
    }

}
