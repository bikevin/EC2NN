package bi.kevin;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by Kevin on 8/23/2016.
 */
public class DataFormatter{

    private int[] labelIndicies;
    private Collection<? extends Collection<Double>> collection;

    public DataFormatter(int[] labelIndicies, Collection<? extends Collection<Double>> collection) throws Exception {

        //allow empty labels - if empty, create a labelless collection

        if(collection.size() < 1 || collection.iterator().next().size() < 1){
            throw new Exception("Size of all collections must be greater than 0.");
        } else if (labelIndicies.length > 0){
            for(int index : labelIndicies){
                if(index < 0 || index >= collection.iterator().next().size()){
                    throw new Exception("Label indicies not contained in the dataset");
                }
            }
        }

        this.labelIndicies = labelIndicies;
        this.collection = collection;
    }

    public DataSet getAllData(){

        //initialize the empty containers
        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for(Collection<Double> record : collection){

            //return a dataset containing one point/row
            DataSet temp = getOneDatum(record);

            //push the individual points to collections
            inputs.add(temp.getFeatureMatrix());
            labels.add(temp.getLabels());
        }

        //put the datasets together into one big dataset
        return new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])),
                Nd4j.vstack(labels.toArray(new INDArray[0])));
    }

    private DataSet getOneDatum(Collection<Double> record){

        List<Double> currentList;

        //convert the collection into an arraylist
        if(record instanceof List){
            currentList = (List<Double>) record;
        } else {
            currentList = new ArrayList<>(record);
        }

        //initialize the INDArrays
        INDArray label = Nd4j.create(labelIndicies.length);
        INDArray features = Nd4j.create(record.size() - labelIndicies.length);

        //separate the arraylist into features and labels
        int featureCount = 0;
        int labelCount = 0;
        for(int i = 0; i < currentList.size(); i++){
            boolean isLabel = false;
            for(int index : labelIndicies){
                if(i == index){
                    isLabel = true;
                }
            }

            if(isLabel){
                label.putScalar(labelCount++, currentList.get(i));
            } else {
                features.putScalar(featureCount++, currentList.get(i));
            }
        }

        //return the dataset with a single point in it
        return new DataSet(features, label);
    }

    public DataSet getAllDataNormalized(){

        //normalize the data
        DataSet unNormalized = getAllData();

        DataNormalization normalizer = new NormalizerStandardize();
        
        //set the normalizer parameters - this doesn't actually normalize the dataset
        normalizer.fit(unNormalized);
        
        //this normalizes the dataset
        normalizer.transform(unNormalized);

        return unNormalized;
    }
}
