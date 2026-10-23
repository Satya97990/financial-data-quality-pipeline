import pandas as pd
import numpy as np

class FinancialDataQualityValidator:
    """
    Executes rule-based and threshold-driven data quality checks on raw market feeds.
    Quarantines hard schema/rule violations before downstream statistical ML modeling.
    """
    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        self.audit_metrics = {}
        
    def run_validation_suite(self):
        df = self.raw_df.copy()
        total_initial = len(df)
        
        # 1. Duplicate Record Check
        dup_mask = df.duplicated(subset=['record_id', 'timestamp', 'ticker'], keep='first')
        n_duplicates = int(dup_mask.sum())
        df = df[~dup_mask].copy()
        
        # 2. Missing Critical Fields Check
        null_mask = df[['trade_price', 'bid_price', 'ask_price', 'volume']].isnull().any(axis=1)
        n_nulls = int(null_mask.sum())
        
        # 3. Non-Positive Price or Volume Check (Hard Domain Rule)
        non_pos_mask = (df['trade_price'] <= 0) | (df['bid_price'] <= 0) | (df['ask_price'] <= 0) | (df['volume'] <= 0)
        n_non_positive = int(non_pos_mask.sum())
        
        # 4. Crossed Market Check (Bid > Ask Inversion)
        crossed_mask = df['bid_price'] > df['ask_price']
        n_crossed = int(crossed_mask.sum())
        
        # Combine hard rule failures to quarantine bad records
        hard_failure_mask = null_mask | non_pos_mask | crossed_mask
        quarantined_df = df[hard_failure_mask].copy()
        clean_df = df[~hard_failure_mask].copy()
        
        self.audit_metrics = {
            'total_records_received': total_initial,
            'duplicate_records_removed': n_duplicates,
            'missing_value_violations': n_nulls,
            'non_positive_value_violations': n_non_positive,
            'crossed_bid_ask_violations': n_crossed,
            'total_quarantined_records': len(quarantined_df),
            'records_passed_to_ml_pipeline': len(clean_df),
            'overall_pass_rate_pct': round((len(clean_df) / total_initial) * 100, 2)
        }
        
        print(f"[Quality Validator] Pass rate: {self.audit_metrics['overall_pass_rate_pct']}% "
              f"({len(quarantined_df)} records quarantined by hard business rules)")
        
        return clean_df.reset_index(drop=True), quarantined_df.reset_index(drop=True), self.audit_metrics