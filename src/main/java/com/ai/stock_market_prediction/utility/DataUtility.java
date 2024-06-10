package com.ai.stock_market_prediction.utility;

import org.nd4j.linalg.dataset.DataSet;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

public class DataUtility {

    private static DataSet loadData(String filePath) throws IOException, InterruptedException {
        DataSet dataSet;
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
            recordReader.initialize(new FileSplit(new File(filePath)));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 10, 1, 2);
            dataSet = iterator.next();
        }
        return dataSet;
    }

    private static void normalizeData(DataSet dataSet) {
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
    }

    private SplitTestAndTrain splitTrainAndTestData(DataSet allData, int splitRatio){
        SplitTestAndTrain splitTestAndTrain = allData.splitTestAndTrain(splitRatio);
        return splitTestAndTrain;
    }
}
