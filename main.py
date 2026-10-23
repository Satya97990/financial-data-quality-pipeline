import os
import pandas as pd
from src.data_generator import generate_market_tick_data
from src.quality_validator import FinancialDataQualityValidator
from src.feature_engineering import build_statistical_features
from src.anomaly_models import AnomalyDetectionSuite
from src.visualizer import generate_visual_reports

def main():
    print("=" * 75)
    print(" FINANCIAL DATA QUALITY & STATISTICAL ANOMALY DETECTION PIPELINE ")
    print("=" * 75)
    
    # Step 1: Acquire / Simulate Market Feed Data
    raw_df = generate_market_tick_data(n_rows=15000, seed=42)
    
    # Step 2: Run Stage-1 Deterministic Data Quality Gate
    validator = FinancialDataQualityValidator(raw_df)
    clean_df, quarantined_df, audit_metrics = validator.run_validation_suite()
    
    # Save quarantined records for operations review
    os.makedirs('outputs', exist_ok=True)
    quarantined_df.to_csv('outputs/quarantined_rule_violations.csv', index=False)
    
    # Step 3: Statistical Feature Engineering & Dimensionality Reduction (PCA)
    featured_df, X_scaled, X_pca, feature_cols = build_statistical_features(clean_df)
    
    # Step 4: Statistical & ML Anomaly Detection (Stage-2 Gate)
    y_true = featured_df['is_injected_anomaly'].values
    contamination_est = max(0.015, round(y_true.mean(), 4))
    
    detector = AnomalyDetectionSuite(X_scaled, y_true, contamination=contamination_est)
    kmeans_preds, _ = detector.run_kmeans_detector(n_clusters=4)
    gmm_preds, gmm_scores = detector.run_gmm_detector(n_components=4)
    metrics_df = detector.run_supervised_classifiers()
    
    featured_df['gmm_anomaly_flag'] = gmm_preds
    featured_df['gmm_anomaly_score'] = gmm_scores
    
    # Export ML-flagged anomalies (e.g., fat-finger spikes & flash volume surges)
    ml_anomalies = featured_df[featured_df['gmm_anomaly_flag'] == 1]
    ml_anomalies.to_csv('outputs/ml_flagged_market_anomalies.csv', index=False)
    
    # Step 5: Generate Visualizations & Final Summary Report
    generate_visual_reports(featured_df, audit_metrics, gmm_preds)
    
    print("\n--- STAGE 1: RULE-BASED DATA QUALITY AUDIT SUMMARY ---")
    for k, v in audit_metrics.items():
        print(f"  {k.replace('_', ' ').title()}: {v}")
        
    print("\n--- STAGE 2: STATISTICAL & ML ANOMALY DETECTION BENCHMARK ---")
    print(metrics_df.to_string())
    
    # Write text summary to outputs/audit_summary_report.txt
    with open('outputs/audit_summary_report.txt', 'w') as f:
        f.write("FINANCIAL DATA QUALITY & ANOMALY DETECTION AUDIT REPORT\n")
        f.write("=======================================================\n\n")
        f.write("1. Deterministic Quality Gate Metrics:\n")
        for k, v in audit_metrics.items():
            f.write(f"   - {k}: {v}\n")
        f.write("\n2. Machine Learning Anomaly Detection Benchmark:\n")
        f.write(metrics_df.to_string())
        f.write("\n")
        
    print("\n[Pipeline Complete] All reports, flagged CSVs, and plots saved in 'outputs/'.")

if __name__ == "__main__":
    main()