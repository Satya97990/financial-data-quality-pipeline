import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_statistical_features(df: pd.DataFrame):
    """
    Computes ticker-grouped statistical features (rolling Z-scores, relative bid-ask spread,
    log volume ratios, price returns) and projects scaled features into 2D PCA space.
    """
    df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    
    # Group by ticker so statistical baselines don't mix different stock price scales
    grouped = df.groupby('ticker')
    
    # Rolling mean and std for trade price (window=30 ticks, fallback to expanding for early rows)
    rolling_mean = grouped['trade_price'].transform(lambda x: x.rolling(30, min_periods=5).mean().bfill())
    rolling_std = grouped['trade_price'].transform(lambda x: x.rolling(30, min_periods=5).std().bfill())
    
    # Feature 1: Price Rolling Z-Score (captures sudden spikes / fat-finger trades)
    df['price_zscore'] = np.abs((df['trade_price'] - rolling_mean) / (rolling_std + 1e-6))
    
    # Feature 2: Relative Bid-Ask Spread in basis points
    mid_price = (df['bid_price'] + df['ask_price']) / 2.0
    df['rel_spread_bps'] = ((df['ask_price'] - df['bid_price']) / (mid_price + 1e-6)) * 10000
    
    # Feature 3: Log Volume Z-Score per ticker
    df['log_volume'] = np.log1p(df['volume'])
    vol_mean = grouped['log_volume'].transform('mean')
    vol_std = grouped['log_volume'].transform('std')
    df['volume_zscore'] = np.abs((df['log_volume'] - vol_mean) / (vol_std + 1e-6))
    
    # Feature 4: Tick-over-Tick Absolute Return (%)
    df['abs_tick_return_pct'] = grouped['trade_price'].transform(
        lambda x: x.pct_change().abs().fillna(0.0) * 100
    )
    
    # Feature 5: Trade Price Deviation from Mid-Quote (%)
    df['mid_quote_deviation_pct'] = (np.abs(df['trade_price'] - mid_price) / (mid_price + 1e-6)) * 100
    
    feature_cols = [
        'price_zscore', 
        'rel_spread_bps', 
        'volume_zscore', 
        'abs_tick_return_pct', 
        'mid_quote_deviation_pct'
    ]
    
    # Clean any infinities and standardize features using NumPy/Scikit-learn
    X_raw = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Apply PCA to project the 5D feature space onto 2 principal components for visualization & clustering
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"[Feature Engineering] Extracted {len(feature_cols)} statistical features. "
          f"2D PCA captures {explained_var:.2f}% variance.")
    
    return df, X_scaled, X_pca, feature_cols