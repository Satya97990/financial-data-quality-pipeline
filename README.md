# Financial Market Data Quality & Statistical Anomaly Detection Pipeline

An end-to-end Python pipeline designed to acquire, validate, model, and monitor high-frequency equity market tick data. Built to catch both deterministic schema/business-rule corruptions and subtle statistical anomalies (such as fat-finger trade spikes and flash volume surges) before downstream ingestion into financial analytics platforms.

## Key Features
1. **Stage-1 Deterministic Data Quality Gate (`src/quality_validator.py`):**
   - Enforces automated validation controls for duplicate transaction IDs, null price/volume fields, non-positive values, and crossed market quotes (`Bid > Ask` inversions).
   - Automatically quarantines corrupted records into `outputs/quarantined_rule_violations.csv` and calculates pipeline pass rates.
2. **Ticker-Grouped Statistical Feature Engineering (`src/feature_engineering.py`):**
   - Computes rolling 30-tick price Z-scores, relative bid-ask spreads (in bps), log-volume Z-scores, and tick-over-tick returns per equity ticker.
   - Applies **Principal Component Analysis (PCA)** to project multi-dimensional market features into 2D space for cluster inspection.
3. **Stage-2 Unsupervised & Supervised ML Anomaly Detection (`src/anomaly_models.py`):**
   - **Unsupervised Detection:** Uses **Gaussian Mixture Models (GMM)** log-likelihood density estimation and **K-Means** centroid distance thresholding to flag anomalous market ticks without requiring labeled historical data.
   - **Supervised Benchmark:** Uses **SMOTE** to address severe class imbalance (~2% anomaly rate) and benchmarks an **RBF-Kernel Support Vector Machine (SVM)** against a **Random Forest** classifier.
4. **Automated Visual & Audit Reporting (`src/visualizer.py`):**
   - Generates high-resolution PCA cluster projections and data quality violation charts in `outputs/`.

## Project Structure
- `src/data_generator.py`: Simulates 15,000+ intra-day equity ticks (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `AAPL.US`, etc.) with injected market anomalies.
- `src/quality_validator.py`: Rule-based validation engine.
- `src/feature_engineering.py`: Rolling statistical feature extraction and PCA reduction.
- `src/anomaly_models.py`: K-Means, GMM, SMOTE, SVM, and Random Forest implementations.
- `src/visualizer.py`: Matplotlib/Seaborn diagnostic plotting suite.
- `main.py`: Master execution script.

## How to Run Locally
```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the end-to-end pipeline
python main.py