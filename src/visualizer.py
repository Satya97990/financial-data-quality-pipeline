import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_visual_reports(df: pd.DataFrame, audit_metrics: dict, gmm_preds):
    """
    Produces two diagnostic charts in 'outputs/':
    1. 2D PCA projection showing normal market ticks vs. GMM-detected statistical anomalies.
    2. Data Quality Audit breakdown of hard-rule violations.
    """
    os.makedirs('outputs', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: 2D PCA Anomaly Cluster Projection ---
    plt.figure(figsize=(10, 6))
    df_plot = df.copy()
    df_plot['Detection_Status'] = ['Flagged Anomaly (GMM)' if p == 1 else 'Normal Market Tick' for p in gmm_preds]
    
    sns.scatterplot(
        data=df_plot, x='pca_1', y='pca_2',
        hue='Detection_Status',
        palette={'Normal Market Tick': '#2b5c8f', 'Flagged Anomaly (GMM)': '#d9534f'},
        alpha=0.7, s=35
    )
    plt.title('2D PCA Projection of Market Tick Features & GMM Anomaly Detection', fontsize=13, fontweight='bold')
    plt.xlabel('Principal Component 1 (Price & Return Volatility)')
    plt.ylabel('Principal Component 2 (Spread & Volume Surge)')
    plt.legend(title='Pipeline Classification')
    plt.tight_layout()
    plt.savefig('outputs/pca_anomaly_projection.png', dpi=300)
    plt.close()
    
    # --- Plot 2: Rule-Based Data Quality Breakdown ---
    violation_counts = {
        'Duplicate Rows': audit_metrics['duplicate_records_removed'],
        'Missing Price/Vol': audit_metrics['missing_value_violations'],
        'Negative/Zero Vol': audit_metrics['non_positive_value_violations'],
        'Crossed Bid-Ask': audit_metrics['crossed_bid_ask_violations']
    }
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(violation_counts.keys(), violation_counts.values(), color=['#4a90e2', '#f5a623', '#d0021b', '#8b572a'])
    plt.title('Stage 1 Data Quality Gate: Quarantined Rule Violations', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Corrupted Records Flagged')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, int(yval), ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('outputs/data_quality_violations.png', dpi=300)
    plt.close()
    print("[Visualizer] Saved diagnostic charts to 'outputs/pca_anomaly_projection.png' and 'outputs/data_quality_violations.png'")