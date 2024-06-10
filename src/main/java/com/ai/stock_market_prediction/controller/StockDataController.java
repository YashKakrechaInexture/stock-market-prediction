package com.ai.stock_market_prediction.controller;

import com.ai.stock_market_prediction.service.StockDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/stockData")
public class StockDataController {

    @Autowired
    private StockDataService stockDataService;

    @PostMapping("/upload")
    public String uploadData(@RequestParam("file") MultipartFile file) {
        try {
            stockDataService.loadData(file);
            return "Data loaded successfully";
        } catch (Exception e) {
            return "Failed to load data: " + e.getMessage();
        }
    }

    @PostMapping("/train")
    public String trainModel() {
        stockDataService.trainModel();
        return "Model trained successfully";
    }

    @GetMapping("/predict")
    public String predict(@RequestParam double open, @RequestParam double close,
                          @RequestParam double high, @RequestParam double low) {
        boolean willBeHigher = stockDataService.predictNextDay(open, close, high, low);
        return willBeHigher ? "The price will go higher" : "The price will go lower";
    }

    @PostMapping("/evaluate")
    public String evaluateModel() {
        return stockDataService.evaluateModel();
    }
}
