"""
Explanation Generator - Generate educational explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """Generate educational explanations for data science concepts"""
    
    def __init__(self):
        pass
    
    def generate_data_analysis_explanation(self, data_info: Dict[str, Any]) -> str:
        """Generate explanation for data analysis results"""
        explanation = "## Data Analysis Results\n\n"
        
        # Shape explanation
        shape = data_info.get('shape', (0, 0))
        explanation += f"**Dataset Overview:**\n"
        explanation += f"- Your dataset has {shape[0]:,} rows and {shape[1]} columns\n"
        explanation += f"- This is a {'small' if shape[0] < 1000 else 'medium' if shape[0] < 10000 else 'large'} dataset\n\n"
        
        # Missing values explanation
        missing_pct = data_info.get('missing_percentage', 0)
        if missing_pct > 0:
            explanation += f"**Missing Values:**\n"
            explanation += f"- {missing_pct:.1f}% of your data is missing\n"
            if missing_pct < 5:
                explanation += "- This is a low missing value rate, which is good!\n"
            elif missing_pct < 20:
                explanation += "- This is a moderate missing value rate that needs attention\n"
            else:
                explanation += "- This is a high missing value rate that requires careful handling\n"
            explanation += "\n"
        
        # Data types explanation
        numeric_cols = len(data_info.get('numeric_columns', []))
        categorical_cols = len(data_info.get('categorical_columns', []))
        
        explanation += f"**Data Types:**\n"
        explanation += f"- {numeric_cols} numeric columns (suitable for mathematical operations)\n"
        explanation += f"- {categorical_cols} categorical columns (need encoding for ML algorithms)\n\n"
        
        return explanation
    
    def generate_preprocessing_explanation(self, step_name: str, data_context: Dict[str, Any]) -> str:
        """Generate explanation for preprocessing steps"""
        explanations = {
            "missing_values": """
**Handling Missing Values**

Missing values can significantly impact your model's performance. Here's why this step is important:

1. **Why it matters**: Most ML algorithms cannot handle missing values directly
2. **Common strategies**:
   - **Mean/Median imputation**: Good for numeric data with normal distribution
   - **Mode imputation**: Best for categorical data
   - **Forward/Backward fill**: Useful for time series data
   - **Model-based imputation**: Advanced technique using other features

3. **Best practices**:
   - Always analyze the pattern of missingness first
   - Consider if missing values are informative (missing at random vs. not at random)
   - Validate your imputation strategy
            """,
            
            "outliers": """
**Outlier Detection and Handling**

Outliers are data points that are significantly different from the rest of your data:

1. **Why detect outliers**:
   - They can skew model training
   - May represent data entry errors
   - Could be genuine but rare events

2. **Detection methods**:
   - **IQR method**: Uses quartiles to identify outliers
   - **Z-score method**: Identifies points beyond 3 standard deviations
   - **Isolation Forest**: Machine learning approach

3. **Handling strategies**:
   - **Cap/Clip**: Limit extreme values to reasonable bounds
   - **Remove**: Delete outlier rows (use carefully)
   - **Transform**: Apply log or other transformations
            """,
            
            "categorical_encoding": """
**Categorical Variable Encoding**

Machine learning algorithms require numeric input, so categorical variables must be converted:

1. **Encoding strategies**:
   - **One-hot encoding**: Creates binary columns for each category
   - **Label encoding**: Assigns numbers to categories
   - **Target encoding**: Uses target variable statistics

2. **When to use each**:
   - One-hot: Nominal categories (no order)
   - Label: Ordinal categories (has order)
   - Target: High cardinality categories

3. **Considerations**:
   - One-hot encoding increases dimensionality
   - Label encoding assumes ordinal relationships
   - Target encoding can cause overfitting
            """,
            
            "scaling": """
**Feature Scaling**

Many algorithms are sensitive to the scale of features:

1. **Why scale features**:
   - Prevents features with large ranges from dominating
   - Improves convergence speed
   - Essential for distance-based algorithms

2. **Scaling methods**:
   - **StandardScaler**: Mean=0, Std=1 (assumes normal distribution)
   - **MinMaxScaler**: Range [0,1] (preserves original distribution)
   - **RobustScaler**: Uses median and IQR (outlier-resistant)

3. **When to scale**:
   - Always for SVM, KNN, Neural Networks
   - Not needed for tree-based algorithms
   - Optional for linear models
            """
        }
        
        return explanations.get(step_name, f"Explanation for {step_name} step.")
    
    def generate_model_explanation(self, model_type: str, performance: Dict[str, Any]) -> str:
        """Generate explanation for model results"""
        explanation = f"## {model_type} Model Results\n\n"
        
        # Performance metrics explanation
        if "accuracy" in performance:
            acc = performance["accuracy"]
            explanation += f"**Accuracy: {acc:.3f}**\n"
            if acc > 0.9:
                explanation += "- Excellent performance! Your model is very accurate.\n"
            elif acc > 0.8:
                explanation += "- Good performance with room for improvement.\n"
            elif acc > 0.7:
                explanation += "- Fair performance, consider feature engineering or different algorithms.\n"
            else:
                explanation += "- Poor performance, review your data and approach.\n"
            explanation += "\n"
        
        if "precision" in performance and "recall" in performance:
            precision = performance["precision"]
            recall = performance["recall"]
            explanation += f"**Precision: {precision:.3f}** - Of all positive predictions, {precision*100:.1f}% were correct\n"
            explanation += f"**Recall: {recall:.3f}** - Of all actual positives, {recall*100:.1f}% were identified\n\n"
        
        # Overfitting/underfitting analysis
        if "train_score" in performance and "test_score" in performance:
            train_score = performance["train_score"]
            test_score = performance["test_score"]
            gap = train_score - test_score
            
            explanation += "**Model Diagnosis:**\n"
            if gap > 0.1:
                explanation += "- Large gap between training and test scores suggests overfitting\n"
                explanation += "- Consider reducing model complexity or adding regularization\n"
            elif gap < 0.02:
                explanation += "- Small gap suggests good generalization\n"
            else:
                explanation += "- Moderate gap, model is learning appropriately\n"
            explanation += "\n"
        
        return explanation
