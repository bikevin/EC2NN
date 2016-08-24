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

        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        for(Collection<Double> record : collection){

            DataSet temp = getOneDatum(record);

            inputs.add(temp.getFeatureMatrix());
            labels.add(temp.getLabels());
        }


        return new DataSet(Nd4j.vstack(inputs.toArray(new INDArray[0])),
                Nd4j.vstack(labels.toArray(new INDArray[0])));
    }

    private DataSet getOneDatum(Collection<Double> record){

        List<Double> currentList;

        if(record instanceof List){
            currentList = (List<Double>) record;
        } else {
            currentList = new ArrayList<>(record);
        }

        INDArray label = Nd4j.create(labelIndicies.length);
        INDArray features = Nd4j.create(record.size() - labelIndicies.length);

        int count = 0;
        for(int i = 0; i < currentList.size(); i++){
            boolean isLabel = false;
            for(int index : labelIndicies){
                if(i == index){
                    isLabel = true;
                }
            }

            if(isLabel){
                label.putScalar(count++, currentList.get(i));
            } else {
                features.putScalar(count++, currentList.get(i));
            }
        }

        return new DataSet(features, label);
    }

    public DataSet getAllDataNormalized(){

        DataSet unNormalized = getAllData();

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(unNormalized);
        normalizer.transform(unNormalized);

        return unNormalized;
    }
}
