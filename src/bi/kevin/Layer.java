package bi.kevin;

public class Layer {
    //1 for Sigmoid, 2 for TanH, 3 for ReLU, 4 for HDF5 data, 0 for undef
    private int layerType = 0;
    private int neurons = 0;
    private int batchSize = 0;
    private String dataFile = "";
    // 0 for undef, 1 for train, 2 for test
    private int phase = 0;
    public Layer(int layerType, int numNeurons){
        this.layerType = layerType;
        neurons = numNeurons;
    }
    public Layer(int layerType, int numNeurons, String file, int batchSize){
        this.layerType = layerType;
        neurons = numNeurons;
        dataFile = file;
        this.batchSize = batchSize;
    }

    public Layer(int layerType, int numNeurons, int phase){
        this.layerType = layerType;

        neurons = numNeurons;
        this.phase = phase;
    }

    public Layer(int layerType, int numNeurons, String file, int phase, int batchSize){
        this.layerType = layerType;
        neurons = numNeurons;
        dataFile = file;
        this.phase = phase;
        this.batchSize = batchSize;
    }

    public int getLayerType(){
        return layerType;
    }

    public int getNeurons(){
        return neurons;
    }

    public String getDataFile(){
        return dataFile;
    }

    public int getPhase(){
        return phase;
    }

    public int getBatchSize() {
        return batchSize;
    }

}


