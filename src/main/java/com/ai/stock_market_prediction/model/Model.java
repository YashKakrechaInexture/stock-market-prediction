package com.ai.stock_market_prediction.model;

import com.ai.stock_market_prediction.entity.StockData;

import java.util.List;

public interface Model {
    void train(List<StockData> data) throws Exception;
    String predict(double open, double close, double high, double low) throws Exception;
    String evaluate() throws Exception;
}
