package com.ai.stock_market_prediction.model;

import com.ai.stock_market_prediction.entity.StockData;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class NeuralNetwork implements Model {

    private MultiLayerNetwork model;

    private DataSet allDataSet;

    private DataSet trainDataSet;

    private DataSet testDataSet;

    @Override
    public void train(List<StockData> data) {
        long nSamples = data.size();
        int inputSize = 4;
        int outputSize = 1;

        INDArray input = Nd4j.zeros(nSamples, inputSize);
        INDArray output = Nd4j.zeros(nSamples, outputSize);

        for (int i=0 ; i<nSamples-1 ; i++) {
            StockData record = data.get(i);
            input.putScalar(new int[]{i, 0}, record.getOpen());
            input.putScalar(new int[]{i, 1}, record.getClose());
            input.putScalar(new int[]{i, 2}, record.getHigh());
            input.putScalar(new int[]{i, 3}, record.getLow());

            StockData tomorrowRecord = data.get(i+1);
            output.putScalar(new int[]{i, 0}, tomorrowRecord.getOpen() > record.getClose() ? 1.0 : 0.0);
        }

        allDataSet = new DataSet(input, output);

        SplitTestAndTrain splitTestAndTrain = allDataSet.splitTestAndTrain(0.8);
        trainDataSet = splitTestAndTrain.getTrain();
        testDataSet = splitTestAndTrain.getTest();
//        NormalizerStandardize normalizer = new NormalizerStandardize();
//        normalizer.fit(dataSet);
//        normalizer.transform(dataSet);

        MultiLayerConfiguration conf = getMultiLayerConfiguration(inputSize, outputSize);

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));
        for(int i=0 ; i<1000 ; i++){
            model.fit(trainDataSet);
        }
    }

    @Override
    public String predict(double open, double close, double high, double low) {
        INDArray input = Nd4j.create(new double[]{open, close, high, low});
        INDArray output = model.output(input);
        System.out.println(output);
        return output.getDouble(0)>0.5 ? "The price will go higher" : "The price will go lower";
    }

    @Override
    public String evaluate() {
        Evaluation evaluation = new Evaluation(2);
        INDArray predictions = model.output(testDataSet.getFeatures());
        evaluation.eval(testDataSet.getLabels(), predictions);
        System.out.println("Accuracy : " + evaluation.stats());
        return evaluation.stats();
    }

    private MultiLayerConfiguration getMultiLayerConfiguration(int inputSize, int outputSize){
        return new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder().nIn(inputSize).nOut(10)
                        .activation(Activation.SOFTMAX).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10)
                        .activation(Activation.SIGMOID).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10)
                        .activation(Activation.SIGMOID).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nIn(10).nOut(outputSize).build())
                .build();
    }
}
