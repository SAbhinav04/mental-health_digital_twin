"""
Feature Drift Monitoring for Mental Health Digital Twin.

Uses the evidently library to detect population-level drift in feature distributions.
Compares baseline (training data) against current (recent realtime data).

Triggered via /api/drift_report/{user_id} endpoint.
Saves HTML reports to reports/ directory.
"""

import pandas as pd
import numpy as np
import logging
import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, Any

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.test_suite import TestSuite
    from evidently.tests import TestNumberOfDriftedFeatures
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("evidently library not installed. Drift detection disabled.")

import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DriftMonitor:
    """
    Monitors feature drift in real-time data using evidently.
    """
    
    def __init__(self, baseline_df: Optional[pd.DataFrame] = None,
                 drift_threshold: float = 0.30,
                 database_path: str = 'realtime_data.db'):
        """
        Args:
            baseline_df: Baseline feature distributions (training data)
            drift_threshold: Alert if >30% of features drift
            database_path: Path to SQLite database
        """
        if not EVIDENTLY_AVAILABLE:
            logging.warning("evidently not available. Drift detection will be limited.")
        
        self.baseline_df = baseline_df
        self.drift_threshold = drift_threshold
        self.database_path = database_path
        self.reports_dir = 'reports'
        Path(self.reports_dir).mkdir(exist_ok=True)
    
    def set_baseline(self, baseline_df: pd.DataFrame):
        """
        Set or update baseline data for drift comparison.
        
        Args:
            baseline_df: Training/reference DataFrame
        """
        self.baseline_df = baseline_df
        logging.info(f"Baseline set with {len(baseline_df)} records")
    
    def get_recent_data(self, user_id: str, days: int = 7) -> Optional[pd.DataFrame]:
        """
        Fetch recent data for a user from SQLite database.
        
        Args:
            user_id: User identifier
            days: Number of recent days to fetch
        
        Returns:
            DataFrame with recent data or None if no data
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_date_str = cutoff_date.isoformat()
            
            query = f"""
            SELECT * FROM time_series 
            WHERE user_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(user_id, cutoff_date_str))
            conn.close()
            
            if df.empty:
                logging.warning(f"No recent data found for user {user_id} in past {days} days")
                return None
            
            logging.info(f"Fetched {len(df)} recent records for user {user_id}")
            return df
        
        except Exception as e:
            logging.error(f"Failed to fetch recent data for user {user_id}: {e}")
            return None
    
    def detect_drift(self, current_df: pd.DataFrame, 
                    feature_cols: Optional[list] = None) -> Dict[str, Any]:
        """
        Detect drift between baseline and current data.
        
        Args:
            current_df: Recent data to check for drift
            feature_cols: Features to monitor (defaults to config.FINAL_MODEL_FEATURES)
        
        Returns:
            Dictionary with drift results
        """
        if self.baseline_df is None:
            logging.warning("No baseline set. Cannot detect drift.")
            return {'error': 'No baseline data'}
        
        if feature_cols is None:
            feature_cols = config.FINAL_MODEL_FEATURES
        
        # Align columns
        available_cols = [col for col in feature_cols if col in current_df.columns]
        
        if not available_cols:
            logging.error(f"No feature columns found in current data")
            return {'error': 'No feature columns found'}
        
        baseline_subset = self.baseline_df[available_cols].copy()
        current_subset = current_df[available_cols].copy()
        
        # Fill NaN
        baseline_subset = baseline_subset.fillna(baseline_subset.mean())
        current_subset = current_subset.fillna(current_subset.mean())
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline_records': len(baseline_subset),
            'current_records': len(current_subset),
            'features_monitored': len(available_cols),
            'drift_detected': False,
            'drifted_features': [],
            'drift_percentage': 0.0,
            'summary': '',
        }
        
        if not EVIDENTLY_AVAILABLE:
            logging.warning("evidently not available. Using statistical drift detection.")
            return self._statistical_drift_detection(baseline_subset, current_subset, results)
        
        try:
            # Use evidently DataDriftPreset
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=baseline_subset, current_data=current_subset)
            
            # Extract drift metrics
            metric_result = report.as_dict()
            
            if 'metrics' in metric_result and len(metric_result['metrics']) > 0:
                drift_metric = metric_result['metrics'][0]
                
                if 'result' in drift_metric and 'drift_by_columns' in drift_metric['result']:
                    drift_by_col = drift_metric['result']['drift_by_columns']
                    
                    drifted = [col for col, is_drift in drift_by_col.items() if is_drift]
                    results['drifted_features'] = drifted
                    results['drift_percentage'] = len(drifted) / len(available_cols) if available_cols else 0
                    
                    if results['drift_percentage'] > self.drift_threshold:
                        results['drift_detected'] = True
                        results['summary'] = (
                            f"⚠️ DRIFT ALERT: {len(drifted)} out of {len(available_cols)} features "
                            f"({results['drift_percentage']*100:.1f}%) show significant drift. "
                            f"Drifted features: {', '.join(drifted[:5])}"
                            + (f" +{len(drifted)-5} more" if len(drifted) > 5 else "")
                        )
                    else:
                        results['summary'] = f"No significant drift detected ({results['drift_percentage']*100:.1f}% features drifted)"
        
        except Exception as e:
            logging.error(f"Evidently drift detection failed: {e}")
            return self._statistical_drift_detection(baseline_subset, current_subset, results)
        
        return results
    
    def _statistical_drift_detection(self, baseline_df: pd.DataFrame, 
                                     current_df: pd.DataFrame,
                                     results: Dict) -> Dict[str, Any]:
        """
        Fallback statistical drift detection using Kolmogorov-Smirnov test.
        """
        from scipy import stats
        
        drifted_features = []
        
        for col in baseline_df.columns:
            try:
                # KS test
                statistic, p_value = stats.ks_2samp(baseline_df[col], current_df[col])
                
                if p_value < 0.05:  # Significant drift at 5% level
                    drifted_features.append(col)
            except Exception as e:
                logging.debug(f"KS test failed for {col}: {e}")
        
        results['drifted_features'] = drifted_features
        results['drift_percentage'] = len(drifted_features) / len(baseline_df.columns)
        
        if results['drift_percentage'] > self.drift_threshold:
            results['drift_detected'] = True
            results['summary'] = (
                f"⚠️ DRIFT ALERT: {len(drifted_features)} out of {len(baseline_df.columns)} features "
                f"({results['drift_percentage']*100:.1f}%) show significant drift."
            )
        else:
            results['summary'] = f"No significant drift detected ({results['drift_percentage']*100:.1f}% features drifted)"
        
        return results
    
    def generate_report(self, user_id: str, days: int = 7,
                       feature_cols: Optional[list] = None) -> Optional[str]:
        """
        Generate comprehensive drift report and save as HTML.
        
        Args:
            user_id: User identifier
            days: Days of recent data to analyze
            feature_cols: Features to monitor
        
        Returns:
            Path to generated report or None
        """
        if self.baseline_df is None:
            logging.warning("No baseline data. Cannot generate report.")
            return None
        
        current_df = self.get_recent_data(user_id, days=days)
        if current_df is None:
            logging.warning(f"No recent data for user {user_id}")
            return None
        
        # Detect drift
        drift_results = self.detect_drift(current_df, feature_cols)
        
        if feature_cols is None:
            feature_cols = config.FINAL_MODEL_FEATURES
        
        available_cols = [col for col in feature_cols if col in current_df.columns]
        
        # Generate HTML report
        html_content = self._generate_html_report(user_id, drift_results, 
                                                  self.baseline_df[available_cols],
                                                  current_df[available_cols])
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.reports_dir, f'drift_{user_id}_{timestamp}.html')
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Drift report saved to {report_path}")
        return report_path
    
    def _generate_html_report(self, user_id: str, drift_results: Dict,
                             baseline_df: pd.DataFrame,
                             current_df: pd.DataFrame) -> str:
        """
        Generate HTML report content.
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drift Report - {user_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; }}
                h1 {{ color: #333; }}
                .alert {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .alert-warning {{ background-color: #fff3cd; border: 1px solid #ffc107; color: #856404; }}
                .alert-success {{ background-color: #d4edda; border: 1px solid #28a745; color: #155724; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .stats {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; margin: 20px 0; }}
                .stat-box {{ padding: 15px; background-color: #f8f9fa; border-radius: 5px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
                .stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Feature Drift Report</h1>
                <p><strong>User ID:</strong> {user_id}</p>
                <p><strong>Generated:</strong> {drift_results.get('timestamp', 'N/A')}</p>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">{drift_results.get('features_monitored', 0)}</div>
                        <div class="stat-label">Features Monitored</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{drift_results.get('baseline_records', 0)}</div>
                        <div class="stat-label">Baseline Records</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{drift_results.get('current_records', 0)}</div>
                        <div class="stat-label">Current Records</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{drift_results.get('drift_percentage', 0)*100:.1f}%</div>
                        <div class="stat-label">Features Drifted</div>
                    </div>
                </div>
                
                <div class="alert {'alert-warning' if drift_results.get('drift_detected') else 'alert-success'}">
                    <strong>{'⚠️ ALERT' if drift_results.get('drift_detected') else '✓ Status'}:</strong> 
                    {drift_results.get('summary', 'No drift detected')}
                </div>
                
                <h2>Drifted Features</h2>
                {"<p>No drift detected.</p>" if not drift_results.get('drifted_features') else 
                 "<ul>" + "".join(f"<li>{f}</li>" for f in drift_results.get('drifted_features', [])) + "</ul>"}
                
                <h2>Feature Statistics</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Baseline Mean</th>
                        <th>Current Mean</th>
                        <th>Baseline Std</th>
                        <th>Current Std</th>
                        <th>Change %</th>
                    </tr>
        """
        
        for col in baseline_df.columns[:20]:  # Limit to first 20 features for brevity
            base_mean = baseline_df[col].mean()
            curr_mean = current_df[col].mean()
            change_pct = ((curr_mean - base_mean) / (abs(base_mean) + 1e-6)) * 100
            
            html += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{base_mean:.4f}</td>
                        <td>{curr_mean:.4f}</td>
                        <td>{baseline_df[col].std():.4f}</td>
                        <td>{current_df[col].std():.4f}</td>
                        <td>{change_pct:.2f}%</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <p style="color: #666; font-size: 12px; margin-top: 30px;">
                    Report generated by Mental Health Digital Twin | Drift Threshold: 30% of features
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_drift_status(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Quick check for drift status without generating full report.
        
        Returns:
            Dictionary with drift status
        """
        current_df = self.get_recent_data(user_id, days=days)
        if current_df is None:
            return {'status': 'no_data', 'message': f'No data for {user_id}'}
        
        drift_results = self.detect_drift(current_df)
        return {
            'status': 'drift_detected' if drift_results.get('drift_detected') else 'normal',
            'drifted_features_count': len(drift_results.get('drifted_features', [])),
            'drift_percentage': drift_results.get('drift_percentage', 0),
            'summary': drift_results.get('summary', ''),
        }


if __name__ == '__main__':
    # Example usage
    logging.info("Drift monitor module ready for import")
