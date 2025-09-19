"""
Preprocessing Pipeline - Main pipeline for data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

from .handlers import MissingValueHandler, OutlierHandler, EncodingHandler
from .transformers import ScalingTransformer, FeatureEngineer
from .validators import DataValidator

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingStep:
    """Represents a single preprocessing step"""
    name: str
    handler: Any
    parameters: Dict[str, Any]
    enabled: bool = True
    description: str = ""

class PreprocessingPipeline:
    """
    Comprehensive preprocessing pipeline with educational guidance
    """
    
    def __init__(self, name: str = "default_pipeline"):
        self.name = name
        self.steps: List[PreprocessingStep] = []
        self.is_fitted = False
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.logs = []
        
    def add_step(self, name: str, handler: Any, parameters: Dict[str, Any] = None, 
                 enabled: bool = True, description: str = "") -> 'PreprocessingPipeline':
        """Add a preprocessing step to the pipeline"""
        step = PreprocessingStep(
            name=name,
            handler=handler,
            parameters=parameters or {},
            enabled=enabled,
            description=description
        )
        self.steps.append(step)
        self._log(f"Added step: {name}")
        return self
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        """Fit the preprocessing pipeline"""
        try:
            self._log("Starting pipeline fitting")
            self.feature_names_in_ = list(X.columns)
            
            # Store original data for reference
            X_processed = X.copy()
            
            for step in self.steps:
                if not step.enabled:
                    self._log(f"Skipping disabled step: {step.name}")
                    continue
                
                self._log(f"Fitting step: {step.name}")
                
                # Fit the step
                if hasattr(step.handler, 'fit'):
                    if y is not None and hasattr(step.handler, 'fit_transform'):
                        X_processed = step.handler.fit_transform(X_processed, y, **step.parameters)
                    else:
                        step.handler.fit(X_processed, y, **step.parameters)
                        X_processed = step.handler.transform(X_processed)
                else:
                    # For custom functions
                    X_processed = step.handler(X_processed, **step.parameters)
                
                self._log(f"Completed step: {step.name}")
            
            self.feature_names_out_ = list(X_processed.columns)
            self.is_fitted = True
            self._log("Pipeline fitting completed successfully")
            
        except Exception as e:
            self._log(f"Error fitting pipeline: {str(e)}", level="ERROR")
            raise
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        try:
            self._log("Starting pipeline transformation")
            X_processed = X.copy()
            
            for step in self.steps:
                if not step.enabled:
                    continue
                
                self._log(f"Transforming with step: {step.name}")
                
                if hasattr(step.handler, 'transform'):
                    X_processed = step.handler.transform(X_processed)
                else:
                    # For custom functions
                    X_processed = step.handler(X_processed, **step.parameters)
            
            self._log("Pipeline transformation completed")
            return X_processed
            
        except Exception as e:
            self._log(f"Error transforming data: {str(e)}", level="ERROR")
            raise
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data in one step"""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names"""
        return self.feature_names_out_ or []
    
    def get_feature_names_in(self) -> List[str]:
        """Get input feature names"""
        return self.feature_names_in_ or []
    
    def get_step_info(self) -> List[Dict[str, Any]]:
        """Get information about all steps"""
        return [
            {
                "name": step.name,
                "enabled": step.enabled,
                "description": step.description,
                "parameters": step.parameters,
                "handler_type": type(step.handler).__name__
            }
            for step in self.steps
        ]
    
    def enable_step(self, step_name: str) -> 'PreprocessingPipeline':
        """Enable a specific step"""
        for step in self.steps:
            if step.name == step_name:
                step.enabled = True
                self._log(f"Enabled step: {step_name}")
                break
        return self
    
    def disable_step(self, step_name: str) -> 'PreprocessingPipeline':
        """Disable a specific step"""
        for step in self.steps:
            if step.name == step_name:
                step.enabled = False
                self._log(f"Disabled step: {step_name}")
                break
        return self
    
    def update_step_parameters(self, step_name: str, parameters: Dict[str, Any]) -> 'PreprocessingPipeline':
        """Update parameters for a specific step"""
        for step in self.steps:
            if step.name == step_name:
                step.parameters.update(parameters)
                self._log(f"Updated parameters for step: {step_name}")
                break
        return self
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'name': self.name,
            'steps': self.steps,
            'is_fitted': self.is_fitted,
            'feature_names_in_': self.feature_names_in_,
            'feature_names_out_': self.feature_names_out_,
            'logs': self.logs
        }
        
        joblib.dump(pipeline_data, filepath)
        self._log(f"Pipeline saved to: {filepath}")
    
    def load_pipeline(self, filepath: str) -> 'PreprocessingPipeline':
        """Load a saved pipeline"""
        pipeline_data = joblib.load(filepath)
        
        self.name = pipeline_data['name']
        self.steps = pipeline_data['steps']
        self.is_fitted = pipeline_data['is_fitted']
        self.feature_names_in_ = pipeline_data['feature_names_in_']
        self.feature_names_out_ = pipeline_data['feature_names_out_']
        self.logs = pipeline_data['logs']
        
        self._log(f"Pipeline loaded from: {filepath}")
        return self
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Add log entry"""
        log_entry = {
            "timestamp": pd.Timestamp.now(),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry)
        
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get pipeline logs"""
        return self.logs
    
    def create_educational_pipeline(self, data: pd.DataFrame, target_column: Optional[str] = None) -> 'PreprocessingPipeline':
        """Create a pipeline with educational steps based on data characteristics"""
        self._log("Creating educational preprocessing pipeline")
        
        # Analyze data characteristics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Remove target column from feature columns
        if target_column and target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        if target_column and target_column in datetime_cols:
            datetime_cols.remove(target_column)
        
        # Step 1: Handle missing values
        if data.isnull().sum().sum() > 0:
            self.add_step(
                name="missing_values",
                handler=MissingValueHandler(),
                parameters={"strategy": "auto"},
                description="Handle missing values using appropriate strategies"
            )
        
        # Step 2: Handle outliers (for numeric columns)
        if numeric_cols:
            self.add_step(
                name="outliers",
                handler=OutlierHandler(),
                parameters={"method": "iqr", "columns": numeric_cols},
                description="Detect and handle outliers in numeric columns"
            )
        
        # Step 3: Encode categorical variables
        if categorical_cols:
            self.add_step(
                name="categorical_encoding",
                handler=EncodingHandler(),
                parameters={"strategy": "auto", "columns": categorical_cols},
                description="Encode categorical variables for machine learning"
            )
        
        # Step 4: Feature engineering
        self.add_step(
            name="feature_engineering",
            handler=FeatureEngineer(),
            parameters={"datetime_columns": datetime_cols},
            description="Create new features from existing data"
        )
        
        # Step 5: Scale features
        if numeric_cols:
            self.add_step(
                name="scaling",
                handler=ScalingTransformer(),
                parameters={"method": "standard", "columns": numeric_cols},
                description="Scale numeric features for algorithms sensitive to scale"
            )
        
        # Step 6: Final validation
        self.add_step(
            name="validation",
            handler=DataValidator(),
            parameters={},
            description="Validate preprocessed data quality"
        )
        
        self._log("Educational pipeline created successfully")
        return self
    
    def get_educational_explanations(self) -> Dict[str, str]:
        """Get educational explanations for each step"""
        explanations = {}
        
        for step in self.steps:
            if step.enabled:
                explanations[step.name] = self._get_step_explanation(step)
        
        return explanations
    
    def _get_step_explanation(self, step: PreprocessingStep) -> str:
        """Get educational explanation for a specific step"""
        explanations = {
            "missing_values": """
            **Handling Missing Values**
            
            Missing values can significantly impact model performance. This step:
            - Identifies patterns in missing data
            - Applies appropriate imputation strategies
            - Preserves data integrity while maximizing information retention
            
            Common strategies:
            - Mean/Median for numeric data
            - Mode for categorical data
            - Forward/Backward fill for time series
            - Model-based imputation for complex patterns
            """,
            
            "outliers": """
            **Outlier Detection and Handling**
            
            Outliers can skew model performance and predictions. This step:
            - Identifies outliers using statistical methods (IQR, Z-score)
            - Provides options to cap, remove, or transform outliers
            - Preserves important information while reducing noise
            
            Why it matters:
            - Prevents outliers from dominating model training
            - Improves model stability and generalization
            - Ensures more robust predictions
            """,
            
            "categorical_encoding": """
            **Categorical Variable Encoding**
            
            Machine learning algorithms require numeric input. This step:
            - Converts categorical data to numeric format
            - Chooses appropriate encoding strategy based on data characteristics
            - Handles high-cardinality categories effectively
            
            Encoding strategies:
            - One-hot encoding for nominal categories
            - Label encoding for ordinal categories
            - Target encoding for high-cardinality categories
            - Embedding for text-like categories
            """,
            
            "feature_engineering": """
            **Feature Engineering**
            
            Creating new features can significantly improve model performance. This step:
            - Generates polynomial features for non-linear relationships
            - Creates interaction terms between important variables
            - Extracts meaningful information from datetime columns
            - Reduces dimensionality through feature selection
            
            Benefits:
            - Captures complex patterns in data
            - Improves model accuracy
            - Reduces overfitting through better feature representation
            """,
            
            "scaling": """
            **Feature Scaling**
            
            Many algorithms are sensitive to feature scale. This step:
            - Normalizes features to similar scales
            - Prevents features with large ranges from dominating
            - Improves convergence speed for iterative algorithms
            
            Scaling methods:
            - StandardScaler: Mean=0, Std=1
            - MinMaxScaler: Range [0,1]
            - RobustScaler: Median and IQR based (outlier-resistant)
            """,
            
            "validation": """
            **Data Validation**
            
            Ensures preprocessed data meets quality standards. This step:
            - Checks for remaining missing values
            - Validates data types and ranges
            - Ensures feature consistency
            - Provides quality metrics and warnings
            
            Quality checks:
            - No infinite values
            - Appropriate data types
            - Reasonable value ranges
            - Feature correlation analysis
            """
        }
        
        return explanations.get(step.name, f"Educational explanation for {step.name} step.")
