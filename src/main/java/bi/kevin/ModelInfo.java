package bi.kevin;

public class ModelInfo {
    private double testScore;
    private double trainScore;
    private int iteration;

    public ModelInfo(double testScore, double trainScore, int iteration){
        this.testScore = testScore;
        this.trainScore = trainScore;
        this.iteration = iteration;
    }

    public void setTestScore(int testScore){
        this.testScore = testScore;
    }

    public void setTrainScore(int trainScore){
        this.trainScore = trainScore;
    }

    public void setIteration(int iteration){
        this.iteration = iteration;
    }

    public double getTestScore(){
        return testScore;
    }

    public double getTrainScore(){
        return trainScore;
    }

    public int getIteration(){
        return iteration;
    }
}
