package bi.kevin;

import com.google.gson.*;
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.lang.reflect.Type;
import java.util.Iterator;

/**
 * Created by Kevin on 8/23/2016.
 */
public class LocalNetwork {

    private DataSet testData;
    private DataSet trainData;
    private int iterations, numInputs, numOutputs;
    private Layer[] layerArray;
    private boolean isClassification;
    private MultiLayerNetwork trainedModel;

    public LocalNetwork(DataSet dataSet, double testTrainSplit, int iterations, Layer[] layerArray, boolean isClassification) throws Exception {
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(testTrainSplit);
        testData = testAndTrain.getTest();
        trainData = testAndTrain.getTrain();
        this.iterations = iterations;
        this.layerArray = layerArray;
        numInputs = dataSet.numInputs();
        numOutputs = dataSet.numOutcomes();
        this.isClassification = isClassification;
        trainedModel = new MultiLayerNetwork(buildNet());
    }

    public LocalNetwork(DataSet dataSet, double testTrainSplit, String json){
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(testTrainSplit);
        testData = testAndTrain.getTest();
        trainData = testAndTrain.getTrain();

        Gson gson = new GsonBuilder().serializeSpecialFloatingPointValues()
                .registerTypeAdapter(ModelState.class, new ModelStateDeserializer())
                .create();

        ModelState modelState = gson.fromJson(json, ModelState.class);

        trainedModel = new MultiLayerNetwork(modelState.getConf().toJson(), new NDArray(new DoubleBuffer(modelState.getParams())));

        numInputs = trainedModel.getLayer(0).getParam("W").rows();
        numOutputs = trainedModel.getLayer(trainedModel.getLayers().length - 1).getParam("W").columns();
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
                .updater(Updater.ADADELTA).rho(0.95)
                .list();

        int layerCount = 0;

        int prevSize;

        for(int i = 1; i < layerArray.length - 1; i++){

            Layer prevLayer = layerArray[i - 1];
            Layer currentLayer = layerArray[i];

            if(currentLayer.getPhase() != 2 && currentLayer.getLayerType() != 4){

                prevSize = prevLayer.getNeurons();

                if(prevLayer.getLayerType() == 4){
                    prevSize = numInputs;
                }

                DenseLayer tempLayer = new DenseLayer.Builder().nIn(prevSize).nOut(currentLayer.getNeurons())
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

    public ArrayList<ModelInfo> trainModel(IterationListener... iterationListener) throws Exception{
        ArrayList<ModelInfo> diagInfo = new ArrayList<>();

        trainedModel.init();
        trainedModel.setListeners(iterationListener);

        trainedModel.fit(trainData);

        return diagInfo;
    }

    public Evaluation getEval(){

        Evaluation eval = new Evaluation(numOutputs);
        INDArray output = trainedModel.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);

        return eval;
    }

    public MultiLayerNetwork getTrainedModel(){
        return trainedModel;
    }

    public DataSet getTestData(){
        return testData;
    }

    public DataSet getTrainData(){
        return trainData;
    }

    public ArrayList< INDArray > getTrainImportance(){

        ArrayList<INDArray> weights = getNetworkWeights();

        ArrayList< INDArray > importances = new ArrayList<>();

        for(int i = 0; i < trainData.numExamples(); i++){
            importances.add(getDatumImportance(weights, trainData.get(i).getFeatures()));
        }

        return importances;
    }

    public ArrayList< INDArray > getTestImportance(){
        ArrayList<INDArray> weights = getNetworkWeights();

        ArrayList< INDArray > importances = new ArrayList<>();

        for(int i = 0; i < testData.numExamples(); i++){
            importances.add(getDatumImportance(weights, testData.get(i).getFeatures()));
        }

        return importances;
    }

    public ArrayList< INDArray > getImportance(DataSet data){
        ArrayList<INDArray> weights = getNetworkWeights();

        ArrayList< INDArray > importances = new ArrayList<>();

        for(int i = 0; i < trainData.numExamples(); i++){
            importances.add(getDatumImportance(weights, data.get(i).getFeatures()));
        }

        return importances;
    }

    private INDArray getDatumImportance(ArrayList<INDArray> weights, INDArray datum){
        ArrayList<INDArray> activations = new ArrayList<>();

        INDArray output = trainedModel.output(datum);

        org.deeplearning4j.nn.api.Layer[] layers = trainedModel.getLayers();

        for(org.deeplearning4j.nn.api.Layer layer : layers){
            activations.add(((BaseLayer) layer).getInput());
        }

        activations.add(output);

        ArrayList<INDArray> refActivations = getRefActivations();

        ArrayList<INDArray> deltaActivations = new ArrayList<>();

        for(int i = 0; i < refActivations.size(); i++){
            deltaActivations.add(activations.get(i).sub(refActivations.get(i)));
        }

        INDArray currentValues = deltaActivations.get(deltaActivations.size() - 1);

        currentValues = currentValues.repeat(0, new int[]{weights.get(weights.size() - 1).rows()});
        currentValues.muli(weights.get(weights.size() - 1));

        for(int i = deltaActivations.size() - 2; i > 0; i--){

            INDArray deltaActRep = deltaActivations.get(i).repeat(1, new int[]{currentValues.columns()});
            currentValues.muli(deltaActRep);

            currentValues = weights.get(i - 1).mmul(currentValues);

        }

        INDArray deltaActRep = deltaActivations.get(0).repeat(1, new int[]{currentValues.columns()});
        currentValues.muli(deltaActRep);

        return currentValues;
    }

    private ArrayList<INDArray> getNetworkWeights(){

        ArrayList<INDArray> weights = new ArrayList<>();

        for (org.deeplearning4j.nn.api.Layer layer : trainedModel.getLayers()){
            weights.add(layer.getParam("W"));
        }

        return weights;
    }

    private ArrayList<INDArray> getRefActivations(){
        ArrayList<INDArray> activations = new ArrayList<>();

        double[] ref = new double[numInputs];

        for(int i = 0; i < numInputs; i++){
            ref[i] = 0;
        }

        INDArray output = trainedModel.output(new NDArray(new DoubleBuffer(ref)));

        org.deeplearning4j.nn.api.Layer[] layers = trainedModel.getLayers();

        for(org.deeplearning4j.nn.api.Layer layer : layers){
            activations.add(((BaseLayer) layer).getInput());
        }

        activations.add(output);

        return activations;
    }

    public String toJson(){
        Gson gson = new GsonBuilder().serializeSpecialFloatingPointValues()
                .registerTypeAdapter(ModelState.class, new ModelStateSerializer())
                .create();

        double[] params = trainedModel.params().data().asDouble();

        ModelState modelState = new ModelState(trainedModel.getLayerWiseConfigurations(), params);

        return gson.toJson(modelState);
    }

    private class ModelState{
        private MultiLayerConfiguration conf;
        private double[] params;

        ModelState(MultiLayerConfiguration conf, double[] params){
            this.conf = conf;
            this.params = params;
        }

        public MultiLayerConfiguration getConf(){
            return conf;
        }

        public double[] getParams(){
            return params;
        }
    }

    private class ModelStateDeserializer implements JsonDeserializer<ModelState> {

        @Override
        public ModelState deserialize(JsonElement jsonElement, Type type, JsonDeserializationContext jsonDeserializationContext) throws JsonParseException {
            JsonElement conf = jsonElement.getAsJsonObject().get("conf").getAsJsonObject();

            JsonArray arr = jsonElement.getAsJsonObject().get("params").getAsJsonArray();

            Iterator<JsonElement> iterator = arr.iterator();

            ArrayList<Double> paramList = new ArrayList<>();

            while(iterator.hasNext()){
                paramList.add(iterator.next().getAsDouble());
            }

            Double[] params = new Double[paramList.size()];

            params = paramList.toArray(params);

            return new ModelState(MultiLayerConfiguration.fromJson(conf.toString()), ArrayUtils.toPrimitive(params));
        }
    }

    private class ModelStateSerializer implements JsonSerializer<ModelState>{

        @Override
        public JsonElement serialize(ModelState modelState, Type type, JsonSerializationContext jsonSerializationContext) {
            JsonObject confObject = new JsonParser().parse(modelState.getConf().toJson()).getAsJsonObject();

            Gson gson = new Gson();

            String paramString = gson.toJson(modelState.getParams());
            JsonArray paramArray = new JsonParser().parse(paramString).getAsJsonArray();

            JsonObject ret = new JsonObject();
            ret.add("conf", confObject);
            ret.add("params", paramArray);

            return ret;
        }
    }
}
