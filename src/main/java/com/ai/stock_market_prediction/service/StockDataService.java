package com.ai.stock_market_prediction.service;

import org.springframework.web.multipart.MultipartFile;

public interface StockDataService {
    void loadData(MultipartFile file) throws Exception;
    void trainModel();
    boolean predictNextDay(double open, double close, double high, double low);
    String evaluateModel();
}
