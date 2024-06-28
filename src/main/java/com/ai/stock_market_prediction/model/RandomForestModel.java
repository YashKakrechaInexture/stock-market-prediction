package com.ai.stock_market_prediction.model;

import com.ai.stock_market_prediction.entity.StockData;
import org.springframework.stereotype.Component;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Component
public class RandomForestModel implements Model {

    private Instances trainData;

    private Instances testData;

    private RandomForest randomForest;

    @Override
    public void train(List<StockData> data) throws Exception {
        Instances wekaData = convertToWekaInstances(data);

        wekaData.randomize(new Random());

        wekaData.setClassIndex(wekaData.numAttributes() - 1);

        int trainSize = (int) Math.round(wekaData.numInstances() * 0.8);
        int testSize = wekaData.numInstances() - trainSize;

        trainData = new Instances(wekaData, 0, trainSize);
        testData = new Instances(wekaData, trainSize, testSize);

        randomForest = new weka.classifiers.trees.RandomForest();
        randomForest.setNumFeatures(100);
        randomForest.setMaxDepth(5);
        randomForest.buildClassifier(trainData);
    }

    @Override
    public String predict(double open, double close, double high, double low) {
        return "";
    }

    @Override
    public String evaluate() throws Exception {
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.evaluateModel(randomForest, testData);

        // Print evaluation stats
        System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
        System.out.println(evaluation.toMatrixString("\nConfusion Matrix\n======\n"));

        return evaluation.toMatrixString("\nConfusion Matrix\n======\n");
    }

    private Instances convertToWekaInstances(List<StockData> stockDataList) {
        // Define Weka attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("open"));
        attributes.add(new Attribute("close"));
        attributes.add(new Attribute("high"));
        attributes.add(new Attribute("low"));
        attributes.add(new Attribute("sharesTraded"));
        attributes.add(new Attribute("turnoverInCr"));

        // Define class attribute with nominal values (up, down)
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("down");
        classValues.add("up");
        attributes.add(new Attribute("class", classValues));

        // Create Instances object
        Instances data = new Instances("StockData", attributes, stockDataList.size());

        // Populate data
        for (int i=0 ; i<stockDataList.size()-5 ; i++) {
            StockData stockData = stockDataList.get(i);
            DenseInstance instance = new DenseInstance(attributes.size());
            instance.setValue(attributes.get(0), stockData.getOpen());
            instance.setValue(attributes.get(1), stockData.getClose());
            instance.setValue(attributes.get(2), stockData.getHigh());
            instance.setValue(attributes.get(3), stockData.getLow());
            instance.setValue(attributes.get(4), stockData.getSharesTraded());
            instance.setValue(attributes.get(5), stockData.getTurnoverInCr());

            // Set class value based on whether the close price is higher than the open price
//            double upsideMove = stockDataList.get(i+5).getHigh() - stockData.getClose();
//            double downsideMove = stockData.getClose() - stockDataList.get(i+5).getLow();
//            String classValue = upsideMove > downsideMove ? "up" : "down";
            String classValue = stockDataList.get(i+5).getClose() > stockData.getClose() ? "up" : "down";
            instance.setValue(attributes.get(6), classValue);

            data.add(instance);
        }

        return data;
    }
}
