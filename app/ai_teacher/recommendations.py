"""
Recommendation Engine - Generate intelligent recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Generate intelligent recommendations for data science workflows"""
    
    def __init__(self):
        pass
    
    def recommend_preprocessing_steps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recommend preprocessing steps based on data characteristics"""
        recommendations = []
        
        # Check for missing values
        missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
        if missing_pct > 5:
            recommendations.append({
                "step": "handle_missing_values",
                "priority": "high" if missing_pct > 20 else "medium",
                "reason": f"Dataset has {missing_pct:.1f}% missing values",
                "suggestion": "Use appropriate imputation strategy based on data type and missing pattern"
            })
        
        # Check for categorical variables
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            recommendations.append({
                "step": "encode_categorical",
                "priority": "high",
                "reason": f"Found {len(categorical_cols)} categorical columns",
                "suggestion": "Encode categorical variables for machine learning algorithms"
            })
        
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        for col in numeric_cols:
            if len(data[col].dropna()) > 10:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(data) * 0.05:  # More than 5% outliers
                    outlier_cols.append(col)
        
        if outlier_cols:
            recommendations.append({
                "step": "handle_outliers",
                "priority": "medium",
                "reason": f"Found outliers in columns: {outlier_cols}",
                "suggestion": "Consider capping, removing, or transforming outliers"
            })
        
        # Check for high cardinality categorical variables
        high_cardinality_cols = []
        for col in categorical_cols:
            if data[col].nunique() > 20:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            recommendations.append({
                "step": "handle_high_cardinality",
                "priority": "medium",
                "reason": f"High cardinality in columns: {high_cardinality_cols}",
                "suggestion": "Consider target encoding or grouping rare categories"
            })
        
        # Check for scaling needs
        if len(numeric_cols) > 0:
            # Check if scaling is needed by comparing ranges
            ranges = data[numeric_cols].max() - data[numeric_cols].min()
            if ranges.max() / ranges.min() > 100:  # Large range difference
                recommendations.append({
                    "step": "scale_features",
                    "priority": "medium",
                    "reason": "Large range differences between numeric features",
                    "suggestion": "Scale features to prevent large values from dominating"
                })
        
        return recommendations
    
    def recommend_models(self, problem_type: str, data_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend models based on problem type and data characteristics"""
        recommendations = []
        
        n_samples = data_characteristics.get('shape', (0, 0))[0]
        n_features = data_characteristics.get('shape', (0, 0))[1]
        missing_pct = data_characteristics.get('missing_percentage', 0)
        
        if problem_type == "classification":
            if n_samples < 1000:
                recommendations.extend([
                    {
                        "model": "Logistic Regression",
                        "reason": "Good for small datasets, interpretable",
                        "pros": ["Fast", "Interpretable", "No hyperparameters"],
                        "cons": ["Linear decision boundary", "Assumes linear relationships"]
                    },
                    {
                        "model": "Decision Tree",
                        "reason": "Good for small datasets, handles non-linear relationships",
                        "pros": ["Interpretable", "Handles non-linear relationships", "No scaling needed"],
                        "cons": ["Prone to overfitting", "High variance"]
                    }
                ])
            else:
                recommendations.extend([
                    {
                        "model": "Random Forest",
                        "reason": "Robust, handles missing values well",
                        "pros": ["Handles missing values", "Feature importance", "Robust to outliers"],
                        "cons": ["Less interpretable", "Can overfit with many trees"]
                    },
                    {
                        "model": "XGBoost",
                        "reason": "High performance, good for structured data",
                        "pros": ["High performance", "Built-in regularization", "Handles missing values"],
                        "cons": ["Many hyperparameters", "Can overfit", "Less interpretable"]
                    }
                ])
        
        elif problem_type == "regression":
            if n_samples < 1000:
                recommendations.extend([
                    {
                        "model": "Linear Regression",
                        "reason": "Simple, interpretable, good baseline",
                        "pros": ["Fast", "Interpretable", "No hyperparameters"],
                        "cons": ["Assumes linear relationships", "Sensitive to outliers"]
                    },
                    {
                        "model": "Ridge Regression",
                        "reason": "Regularized linear model, prevents overfitting",
                        "pros": ["Regularization", "Handles multicollinearity", "Stable"],
                        "cons": ["Still linear", "Requires hyperparameter tuning"]
                    }
                ])
            else:
                recommendations.extend([
                    {
                        "model": "Random Forest Regressor",
                        "reason": "Robust, handles non-linear relationships",
                        "pros": ["Handles non-linear relationships", "Feature importance", "Robust"],
                        "cons": ["Less interpretable", "Can overfit"]
                    },
                    {
                        "model": "XGBoost Regressor",
                        "reason": "High performance, good for structured data",
                        "pros": ["High performance", "Built-in regularization", "Handles missing values"],
                        "cons": ["Many hyperparameters", "Can overfit"]
                    }
                ])
        
        elif problem_type == "clustering":
            recommendations.extend([
                {
                    "model": "K-Means",
                    "reason": "Simple, fast, good for spherical clusters",
                    "pros": ["Fast", "Simple", "Scalable"],
                    "cons": ["Assumes spherical clusters", "Requires k to be specified"]
                },
                {
                    "model": "DBSCAN",
                    "reason": "Finds clusters of arbitrary shape, handles noise",
                    "pros": ["Finds arbitrary shapes", "Handles noise", "No need to specify k"],
                    "cons": ["Sensitive to parameters", "Can struggle with varying densities"]
                }
            ])
        
        return recommendations
    
    def recommend_feature_engineering(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recommend feature engineering steps"""
        recommendations = []
        
        # Check for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            recommendations.append({
                "step": "extract_datetime_features",
                "reason": f"Found {len(datetime_cols)} datetime columns",
                "suggestion": "Extract year, month, day, dayofweek, hour from datetime columns"
            })
        
        # Check for high correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recommendations.append({
                    "step": "handle_multicollinearity",
                    "reason": f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                    "suggestion": "Consider removing one of each highly correlated pair or use PCA"
                })
        
        # Check for polynomial features
        if len(numeric_cols) > 0:
            recommendations.append({
                "step": "create_polynomial_features",
                "reason": "Numeric features available for polynomial expansion",
                "suggestion": "Create polynomial features to capture non-linear relationships"
            })
        
        # Check for interaction features
        if len(numeric_cols) > 1:
            recommendations.append({
                "step": "create_interaction_features",
                "reason": "Multiple numeric features available",
                "suggestion": "Create interaction features between important variables"
            })
        
        return recommendations
    
    def recommend_validation_strategy(self, data: pd.DataFrame, problem_type: str) -> Dict[str, Any]:
        """Recommend validation strategy"""
        n_samples = len(data)
        
        if n_samples < 100:
            return {
                "strategy": "simple_split",
                "test_size": 0.2,
                "reason": "Small dataset, use simple train-test split",
                "suggestion": "Consider using all data for training with cross-validation"
            }
        elif n_samples < 1000:
            return {
                "strategy": "cross_validation",
                "cv_folds": 5,
                "reason": "Medium dataset, use cross-validation for better estimates",
                "suggestion": "Use 5-fold cross-validation for robust performance estimates"
            }
        else:
            return {
                "strategy": "stratified_split",
                "test_size": 0.2,
                "cv_folds": 5,
                "reason": "Large dataset, use stratified split with cross-validation",
                "suggestion": "Use stratified split to maintain class distribution, then cross-validation"
            }
