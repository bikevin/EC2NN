package bi.kevin;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;

/**
 * Created by Kevin on 8/23/2016.
 */
public class CustomListener implements IterationListener {

    private DataSet trainData, testData;
    private int printIterations;
    private boolean invoked = false;
    private ArrayList<ModelInfo> infoStore;

    public CustomListener(DataSet trainData, DataSet testData, int printIterations, ArrayList<ModelInfo> infoStore){
        this.trainData = trainData;
        this.testData = testData;
        this.printIterations = printIterations;
        this.infoStore = infoStore;
    }

    @Override
    public boolean invoked(){
        return invoked;
    }

    @Override
    public void invoke(){
        this.invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration){

        if(iteration % printIterations == 0) {

            if (model instanceof MultiLayerNetwork) {

                MultiLayerNetwork newModel = (MultiLayerNetwork) model;

                infoStore.add(new ModelInfo(newModel.score(trainData), newModel.score(testData), iteration));

            } else {

                infoStore.add(new ModelInfo(model.score(), model.score(), iteration));

            }
        }


    }


}

