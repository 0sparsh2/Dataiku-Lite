"""
Preprocessing Transformers - Additional transformation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings

logger = logging.getLogger(__name__)

class ScalingTransformer:
    """Enhanced scaling transformer with multiple methods"""
    
    def __init__(self, method: str = "standard", columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns
        self.scalers = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'ScalingTransformer':
        """Fit the scaling transformer"""
        try:
            self.is_fitted = True
            
            # Use specified columns or all numeric columns
            if self.columns is None:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
            else:
                numeric_cols = [col for col in self.columns if col in X.columns]
            
            for column in numeric_cols:
                if self.method == "standard":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                elif self.method == "minmax":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                elif self.method == "robust":
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                elif self.method == "normalize":
                    from sklearn.preprocessing import Normalizer
                    scaler = Normalizer()
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                
                scaler.fit(X[[column]])
                self.scalers[column] = scaler
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting ScalingTransformer: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by scaling features"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        try:
            X_processed = X.copy()
            
            for column, scaler in self.scalers.items():
                X_processed[column] = scaler.transform(X[[column]]).flatten()
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error transforming with ScalingTransformer: {e}")
            raise

class FeatureEngineer:
    """Advanced feature engineering utilities"""
    
    def __init__(self, datetime_columns: Optional[List[str]] = None, 
                 create_polynomial: bool = True, 
                 create_interactions: bool = True):
        self.datetime_columns = datetime_columns or []
        self.create_polynomial = create_polynomial
        self.create_interactions = create_interactions
        self.polynomial_features = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'FeatureEngineer':
        """Fit the feature engineer"""
        try:
            self.is_fitted = True
            
            # Fit polynomial features if enabled
            if self.create_polynomial:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.polynomial_features = PolynomialFeatures(
                        degree=2, 
                        include_bias=False, 
                        interaction_only=True
                    )
                    self.polynomial_features.fit(X[numeric_cols])
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting FeatureEngineer: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by engineering new features"""
        try:
            X_processed = X.copy()
            
            # Extract datetime features
            for column in self.datetime_columns:
                if column in X_processed.columns:
                    dt_series = pd.to_datetime(X_processed[column], errors='coerce')
                    X_processed[f"{column}_year"] = dt_series.dt.year
                    X_processed[f"{column}_month"] = dt_series.dt.month
                    X_processed[f"{column}_day"] = dt_series.dt.day
                    X_processed[f"{column}_dayofweek"] = dt_series.dt.dayofweek
                    X_processed[f"{column}_hour"] = dt_series.dt.hour
                    X_processed[f"{column}_quarter"] = dt_series.dt.quarter
                    X_processed[f"{column}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6])
            
            # Create polynomial features
            if self.create_polynomial and self.polynomial_features is not None:
                numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    poly_features = self.polynomial_features.transform(X_processed[numeric_cols])
                    poly_columns = self.polynomial_features.get_feature_names_out(numeric_cols)
                    poly_df = pd.DataFrame(poly_features, columns=poly_columns, index=X_processed.index)
                    
                    # Add only interaction features (degree=2, interaction_only=True)
                    interaction_cols = [col for col in poly_df.columns if '_' in col]
                    if interaction_cols:
                        X_processed = pd.concat([X_processed, poly_df[interaction_cols]], axis=1)
            
            # Create custom interaction features
            if self.create_interactions:
                X_processed = self._create_interaction_features(X_processed)
            
            # Create statistical features
            X_processed = self._create_statistical_features(X_processed)
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between highly correlated variables"""
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return X
            
            # Calculate correlations
            corr_matrix = X[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Create interaction features for highly correlated pairs
            for col1, col2 in high_corr_pairs[:3]:  # Limit to top 3 pairs
                X_processed[f"{col1}_x_{col2}"] = X_processed[col1] * X_processed[col2]
                X_processed[f"{col1}_div_{col2}"] = X_processed[col1] / (X_processed[col2] + 1e-8)  # Avoid division by zero
            
            return X_processed
            
        except Exception as e:
            logger.warning(f"Error creating interaction features: {e}")
            return X
    
    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return X
            
            # Row-wise statistics
            X['numeric_mean'] = X[numeric_cols].mean(axis=1)
            X['numeric_std'] = X[numeric_cols].std(axis=1)
            X['numeric_max'] = X[numeric_cols].max(axis=1)
            X['numeric_min'] = X[numeric_cols].min(axis=1)
            X['numeric_range'] = X['numeric_max'] - X['numeric_min']
            
            # Count of non-zero values
            X['non_zero_count'] = (X[numeric_cols] != 0).sum(axis=1)
            
            return X
            
        except Exception as e:
            logger.warning(f"Error creating statistical features: {e}")
            return X

class DataValidator:
    """Enhanced data validation with comprehensive checks"""
    
    def __init__(self):
        self.is_fitted = True
        self.validation_rules = {
            "check_infinite": True,
            "check_nan": True,
            "check_duplicates": True,
            "check_data_types": True,
            "check_value_ranges": True
        }
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'DataValidator':
        """Fit the data validator (no-op)"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data and return validation report"""
        try:
            validation_report = self._validate_data(X)
            
            # Log validation results
            for check, result in validation_report.items():
                if not result["passed"]:
                    logger.warning(f"Validation failed: {check} - {result['message']}")
                else:
                    logger.info(f"Validation passed: {check}")
            
            return X
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return X
    
    def _validate_data(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Perform comprehensive data validation"""
        validation_report = {}
        
        # Check for infinite values
        if self.validation_rules["check_infinite"]:
            infinite_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            validation_report["check_infinite"] = {
                "passed": infinite_count == 0,
                "message": f"Found {infinite_count} infinite values" if infinite_count > 0 else "No infinite values",
                "count": infinite_count
            }
        
        # Check for NaN values
        if self.validation_rules["check_nan"]:
            nan_count = X.isnull().sum().sum()
            validation_report["check_nan"] = {
                "passed": nan_count == 0,
                "message": f"Found {nan_count} NaN values" if nan_count > 0 else "No NaN values",
                "count": nan_count
            }
        
        # Check for duplicates
        if self.validation_rules["check_duplicates"]:
            duplicate_count = X.duplicated().sum()
            validation_report["check_duplicates"] = {
                "passed": duplicate_count == 0,
                "message": f"Found {duplicate_count} duplicate rows" if duplicate_count > 0 else "No duplicate rows",
                "count": duplicate_count
            }
        
        # Check data types
        if self.validation_rules["check_data_types"]:
            mixed_types = []
            for col in X.columns:
                if X[col].dtype == 'object':
                    # Check if column has mixed types
                    try:
                        pd.to_numeric(X[col], errors='raise')
                    except:
                        mixed_types.append(col)
            
            validation_report["check_data_types"] = {
                "passed": len(mixed_types) == 0,
                "message": f"Found {len(mixed_types)} columns with mixed types: {mixed_types}" if mixed_types else "All columns have consistent data types",
                "mixed_type_columns": mixed_types
            }
        
        # Check value ranges for numeric columns
        if self.validation_rules["check_value_ranges"]:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            range_issues = []
            
            for col in numeric_cols:
                if X[col].min() < -1e10 or X[col].max() > 1e10:
                    range_issues.append(col)
            
            validation_report["check_value_ranges"] = {
                "passed": len(range_issues) == 0,
                "message": f"Found {len(range_issues)} columns with extreme values: {range_issues}" if range_issues else "All numeric columns have reasonable value ranges",
                "extreme_value_columns": range_issues
            }
        
        return validation_report
