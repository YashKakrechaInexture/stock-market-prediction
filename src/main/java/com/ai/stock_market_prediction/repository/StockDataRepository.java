package com.ai.stock_market_prediction.repository;

import com.ai.stock_market_prediction.entity.StockData;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface StockDataRepository extends JpaRepository<StockData, Long> {

    List<StockData> findAllByOrderByDateAsc();
}
