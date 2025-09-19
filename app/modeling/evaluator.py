"""
Model Evaluator - Model evaluation and metrics utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    classification_report
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities"""
    
    def __init__(self):
        pass
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate classification model performance"""
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f1_score": f1_score(y_true, y_pred, average='weighted')
            }
            
            # Add ROC AUC if probabilities are available
            if y_prob is not None:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
                else:  # Multiclass
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            
            # Confusion matrix
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
            
            # Classification report
            metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {"error": str(e)}
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate regression model performance"""
        try:
            metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
                "mape": self._calculate_mape(y_true, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            return {"error": str(e)}
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate clustering model performance"""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            metrics = {
                "silhouette_score": silhouette_score(X, labels),
                "calinski_harabasz_score": calinski_harabasz_score(X, labels),
                "davies_bouldin_score": davies_bouldin_score(X, labels),
                "n_clusters": len(np.unique(labels)),
                "n_noise": np.sum(labels == -1) if -1 in labels else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating clustering model: {e}")
            return {"error": str(e)}
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        try:
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except:
            return np.inf
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]], 
                      metric: str = "accuracy") -> pd.DataFrame:
        """Compare multiple models based on a metric"""
        try:
            comparison_data = []
            
            for model_name, results in model_results.items():
                if metric in results:
                    comparison_data.append({
                        "Model": model_name,
                        "Metric": results[metric],
                        "Type": results.get("type", "unknown")
                    })
            
            return pd.DataFrame(comparison_data).sort_values("Metric", ascending=False)
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()
    
    def generate_evaluation_report(self, metrics: Dict[str, Any], 
                                 model_name: str = "Model") -> str:
        """Generate a human-readable evaluation report"""
        try:
            report = f"Evaluation Report for {model_name}\n"
            report += "=" * 50 + "\n\n"
            
            # Basic metrics
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    report += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
            
            # Confusion matrix
            if "confusion_matrix" in metrics:
                report += "\nConfusion Matrix:\n"
                cm = np.array(metrics["confusion_matrix"])
                report += str(cm) + "\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {e}")
            return f"Error generating report: {str(e)}"
