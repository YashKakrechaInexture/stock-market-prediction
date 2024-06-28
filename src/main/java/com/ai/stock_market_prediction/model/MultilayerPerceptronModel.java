package com.ai.stock_market_prediction.model;

import com.ai.stock_market_prediction.entity.StockData;
import org.springframework.stereotype.Component;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

@Component
public class MultilayerPerceptronModel implements Model {

    private Instances trainData;

    private Instances testData;

    private MultilayerPerceptron mlp;

    @Override
    public void train(List<StockData> data) throws Exception {
        Instances wekaData = convertToWekaInstances(data);

        Normalize normalize = new Normalize();
        normalize.setInputFormat(wekaData);
        Instances normalizedData = Filter.useFilter(wekaData, normalize);
        wekaData.setClassIndex(wekaData.numAttributes() - 1);

        int trainSize = (int) Math.round(wekaData.numInstances() * 0.8);
        int testSize = wekaData.numInstances() - trainSize;

        trainData = new Instances(wekaData, 0, trainSize);
        testData = new Instances(wekaData, trainSize, testSize);
        mlp = new MultilayerPerceptron();
        mlp.setLearningRate(0.1);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(200);
        mlp.setHiddenLayers("a"); // Auto setup of hidden layers

        mlp.buildClassifier(trainData);
    }

    @Override
    public String predict(double open, double close, double high, double low) throws Exception {
        Enumeration<Attribute> attributeEnumeration = trainData.enumerateAttributes();

        Instance instance = new DenseInstance(5);
        instance.setValue(attributeEnumeration.nextElement(), open);
        instance.setValue(attributeEnumeration.nextElement(), close);
        instance.setValue(attributeEnumeration.nextElement(), high);
        instance.setValue(attributeEnumeration.nextElement(), low);
        instance.setDataset(trainData); // Associate instance with dataset

        // Return the classification result
        double prediction = mlp.classifyInstance(instance);
        System.out.println("Prediction: " + prediction);

        // Convert numeric prediction to class label
        return trainData.classAttribute().value((int) prediction);
    }

    @Override
    public String evaluate() throws Exception {
        // Cross-validation for better evaluation
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.crossValidateModel(mlp, trainData, 10, new Debug.Random(1));

        // Evaluate on test data
        evaluation.evaluateModel(mlp, testData);

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

            // Set class value based on whether the close price is higher than the open price
//            double upsideMove = stockDataList.get(i+5).getHigh() - stockData.getClose();
//            double downsideMove = stockData.getClose() - stockDataList.get(i+5).getLow();
//            String classValue = upsideMove > downsideMove ? "up" : "down";
            String classValue = stockDataList.get(i+5).getClose() > stockData.getClose() ? "up" : "down";
            instance.setValue(attributes.get(4), classValue);

            data.add(instance);
        }

        return data;
    }
}
