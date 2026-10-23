import os
import numpy as np
import pandas as pd

def generate_market_tick_data(n_rows=15000, seed=42):
    """
    Simulates intra-day equity market feed data for Indian & Global tickers.
    Intentionally injects realistic data quality flaws and market anomalies 
    (fat-finger trades, stale feeds, negative volume, bid-ask inversions)
    to test the validation and anomaly detection pipeline.
    """
    np.random.seed(seed)
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'AAPL.US', 'MSFT.US']
    base_prices = {
        'RELIANCE.NS': 2950.0, 
        'TCS.NS': 4100.0, 
        'HDFCBANK.NS': 1650.0, 
        'INFY.NS': 1850.0,
        'AAPL.US': 220.0,
        'MSFT.US': 430.0
    }
    
    records = []
    start_time = pd.Timestamp('2026-10-01 09:15:00')
    
    for i in range(n_rows):
        ticker = np.random.choice(tickers)
        base = base_prices[ticker]
        
        # Simulate slight random walk noise around base price
        price_noise = np.random.normal(0, base * 0.005)
        close_price = round(base + price_noise, 2)
        
        spread = round(np.random.uniform(0.05, base * 0.002), 2)
        bid_price = round(close_price - spread / 2, 2)
        ask_price = round(close_price + spread / 2, 2)
        
        volume = int(np.random.lognormal(mean=7.5, sigma=1.2))
        trade_time = start_time + pd.Timedelta(seconds=i * np.random.randint(1, 4))
        
        records.append({
            'record_id': f"TXN_{100000 + i}",
            'timestamp': trade_time,
            'ticker': ticker,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'trade_price': close_price,
            'volume': volume,
            'is_injected_anomaly': 0,
            'anomaly_type': 'Normal'
        })
        
    df = pd.DataFrame(records)
    
    # --- Inject Realistic Data Quality Issues & Market Anomalies (~3.5% of dataset) ---
    n_anomalies = int(n_rows * 0.035)
    anomaly_indices = np.random.choice(df.index, size=n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        issue_type = np.random.choice([
            'fat_finger_spike', 
            'crossed_market_spread', 
            'negative_or_zero_volume', 
            'missing_price', 
            'flash_volume_surge'
        ], p=[0.30, 0.25, 0.15, 0.10, 0.20])
        
        df.loc[idx, 'is_injected_anomaly'] = 1
        df.loc[idx, 'anomaly_type'] = issue_type
        
        if issue_type == 'fat_finger_spike':
            # Sudden 15x or 0.1x price entry error
            multiplier = np.random.choice([0.12, 8.5, 12.0])
            df.loc[idx, 'trade_price'] = round(df.loc[idx, 'trade_price'] * multiplier, 2)
            
        elif issue_type == 'crossed_market_spread':
            # Bid price erroneously higher than Ask price
            df.loc[idx, 'bid_price'] = df.loc[idx, 'ask_price'] + round(np.random.uniform(5.0, 25.0), 2)
            
        elif issue_type == 'negative_or_zero_volume':
            # Corrupted feed sending negative or zero volume
            df.loc[idx, 'volume'] = np.random.choice([-500, -10, 0])
            
        elif issue_type == 'missing_price':
            # Null value in critical field
            df.loc[idx, 'trade_price'] = np.nan
            
        elif issue_type == 'flash_volume_surge':
            # Extreme volume spike accompanied by sharp price drift
            df.loc[idx, 'volume'] = df.loc[idx, 'volume'] * np.random.randint(35, 80)
            df.loc[idx, 'trade_price'] = round(df.loc[idx, 'trade_price'] * np.random.choice([1.18, 0.82]), 2)

    # Inject a few exact duplicate rows to test deduplication checks
    duplicates = df.sample(n=25, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_market_feed.csv', index=False)
    print(f"[Data Generator] Created {len(df)} market tick records in 'data/raw_market_feed.csv'")
    return df