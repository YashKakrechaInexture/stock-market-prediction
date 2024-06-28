package com.ai.stock_market_prediction.service.impl;

import au.com.bytecode.opencsv.CSVReader;
import com.ai.stock_market_prediction.entity.StockData;
import com.ai.stock_market_prediction.model.Model;
import com.ai.stock_market_prediction.repository.StockDataRepository;
import com.ai.stock_market_prediction.service.StockDataService;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
//import org.nd4j.evaluation.classification.Evaluation;
import org.springframework.beans.factory.annotation.Qualifier;
import weka.classifiers.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

@Service
public class StockDataServiceImpl implements StockDataService {

    @Autowired
    private StockDataRepository stockDataRepository;

    @Autowired
    @Qualifier("multilayerPerceptronModel")
    private Model model;


    @Override
    public void loadData(MultipartFile file) throws Exception {
        List<StockData> stockDataList = new ArrayList<>();
        CSVReader reader = new CSVReader(new InputStreamReader(file.getInputStream()));
        String[] currLine = reader.readNext(); // Skip header
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd-MMM-yyyy", Locale.ENGLISH);

        while ((currLine = reader.readNext()) != null) {
            try{
                StockData stockData = new StockData();
                currLine[0] = currLine[0].toLowerCase();
                currLine[0] = currLine[0].substring(0,4).toUpperCase() + currLine[0].substring(4);
                stockData.setDate(LocalDate.parse(currLine[0], formatter));
                stockData.setOpen(Double.parseDouble(currLine[1]));
                stockData.setHigh(Double.parseDouble(currLine[2]));
                stockData.setLow(Double.parseDouble(currLine[3]));
                stockData.setClose(Double.parseDouble(currLine[4]));
                stockData.setSharesTraded(Long.parseLong(currLine[5]));
                stockData.setTurnoverInCr(Double.parseDouble(currLine[6]));
                stockDataList.add(stockData);
            }catch(Exception e){
                System.out.println(e);
            }
        }
        stockDataRepository.saveAll(stockDataList);
    }

    @Override
    public void trainModel() throws Exception {
        List<StockData> data = stockDataRepository.findAllByOrderByDateAsc();
        model.train(data);
    }

    @Override
    public String predictNextDay(double open, double close, double high, double low) throws Exception {
        return model.predict(open, close, high, low);
    }

    @Override
    public String evaluateModel() throws Exception {
        return model.evaluate();
    }


}
