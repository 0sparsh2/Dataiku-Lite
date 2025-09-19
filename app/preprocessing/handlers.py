"""
Preprocessing Handlers - Individual preprocessing components
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class MissingValueHandler:
    """Handle missing values with various strategies"""
    
    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self.imputers = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'MissingValueHandler':
        """Fit the missing value handler"""
        try:
            self.is_fitted = True
            
            for column in X.columns:
                if X[column].isnull().sum() > 0:
                    if X[column].dtype in ['int64', 'float64']:
                        # Numeric column
                        if self.strategy == "auto":
                            strategy = "mean" if X[column].skew() < 2 else "median"
                        else:
                            strategy = self.strategy
                        
                        self.imputers[column] = SimpleImputer(strategy=strategy)
                        self.imputers[column].fit(X[[column]])
                    else:
                        # Categorical column
                        self.imputers[column] = SimpleImputer(strategy="most_frequent")
                        self.imputers[column].fit(X[[column]])
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting MissingValueHandler: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by imputing missing values"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        try:
            X_processed = X.copy()
            
            for column, imputer in self.imputers.items():
                X_processed[column] = imputer.transform(X[[column]]).flatten()
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error transforming with MissingValueHandler: {e}")
            raise

class OutlierHandler:
    """Handle outliers using various detection methods"""
    
    def __init__(self, method: str = "iqr", columns: Optional[List[str]] = None):
        self.method = method
        self.columns = columns
        self.outlier_info = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'OutlierHandler':
        """Fit the outlier handler"""
        try:
            self.is_fitted = True
            
            # Use specified columns or all numeric columns
            if self.columns is None:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
            else:
                numeric_cols = [col for col in self.columns if col in X.columns]
            
            for column in numeric_cols:
                if self.method == "iqr":
                    Q1 = X[column].quantile(0.25)
                    Q3 = X[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = (X[column] < lower_bound) | (X[column] > upper_bound)
                    
                elif self.method == "zscore":
                    z_scores = np.abs(stats.zscore(X[column].dropna()))
                    outliers = z_scores > 3
                    
                elif self.method == "isolation_forest":
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso_forest.fit_predict(X[[column]]) == -1
                
                self.outlier_info[column] = {
                    "outlier_mask": outliers,
                    "outlier_count": outliers.sum(),
                    "outlier_percentage": (outliers.sum() / len(X)) * 100
                }
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting OutlierHandler: {e}")
            raise
    
    def transform(self, X: pd.DataFrame, action: str = "cap") -> pd.DataFrame:
        """Transform data by handling outliers"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        try:
            X_processed = X.copy()
            
            for column, info in self.outlier_info.items():
                if info["outlier_count"] > 0:
                    if action == "cap":
                        # Cap outliers at 95th percentile
                        upper_cap = X[column].quantile(0.95)
                        lower_cap = X[column].quantile(0.05)
                        X_processed[column] = X_processed[column].clip(lower_cap, upper_cap)
                    elif action == "remove":
                        # Remove outlier rows
                        X_processed = X_processed[~info["outlier_mask"]]
                    elif action == "log":
                        # Log transform
                        X_processed[column] = np.log1p(X_processed[column])
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error transforming with OutlierHandler: {e}")
            raise

class EncodingHandler:
    """Handle categorical variable encoding"""
    
    def __init__(self, strategy: str = "auto", columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.columns = columns
        self.encoders = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'EncodingHandler':
        """Fit the encoding handler"""
        try:
            self.is_fitted = True
            
            # Use specified columns or all categorical columns
            if self.columns is None:
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            else:
                categorical_cols = [col for col in self.columns if col in X.columns]
            
            for column in categorical_cols:
                unique_count = X[column].nunique()
                
                if self.strategy == "auto":
                    if unique_count <= 10:
                        # Use one-hot encoding for low cardinality
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoder.fit(X[[column]])
                    else:
                        # Use label encoding for high cardinality
                        encoder = LabelEncoder()
                        encoder.fit(X[column])
                elif self.strategy == "onehot":
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(X[[column]])
                elif self.strategy == "label":
                    encoder = LabelEncoder()
                    encoder.fit(X[column])
                
                self.encoders[column] = {
                    "encoder": encoder,
                    "method": self.strategy,
                    "unique_count": unique_count
                }
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting EncodingHandler: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by encoding categorical variables"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        try:
            X_processed = X.copy()
            
            for column, info in self.encoders.items():
                encoder = info["encoder"]
                method = info["method"]
                
                if method in ["auto", "onehot"] and hasattr(encoder, "transform"):
                    # One-hot encoding
                    encoded = encoder.transform(X[[column]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{column}_{cat}" for cat in encoder.categories_[0]]
                    )
                    X_processed = pd.concat([X_processed.drop(columns=[column]), encoded_df], axis=1)
                elif method in ["auto", "label"] and hasattr(encoder, "transform"):
                    # Label encoding
                    X_processed[column] = encoder.transform(X[column])
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error transforming with EncodingHandler: {e}")
            raise

class ScalingTransformer:
    """Scale numeric features"""
    
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
                    scaler = StandardScaler()
                elif self.method == "minmax":
                    scaler = MinMaxScaler()
                else:
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
    """Feature engineering utilities"""
    
    def __init__(self, datetime_columns: Optional[List[str]] = None):
        self.datetime_columns = datetime_columns or []
        self.is_fitted = True  # No fitting needed for feature engineering
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'FeatureEngineer':
        """Fit the feature engineer (no-op)"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by engineering new features"""
        try:
            X_processed = X.copy()
            
            # Extract datetime features
            for column in self.datetime_columns:
                if column in X_processed.columns:
                    X_processed[f"{column}_year"] = pd.to_datetime(X_processed[column]).dt.year
                    X_processed[f"{column}_month"] = pd.to_datetime(X_processed[column]).dt.month
                    X_processed[f"{column}_day"] = pd.to_datetime(X_processed[column]).dt.day
                    X_processed[f"{column}_dayofweek"] = pd.to_datetime(X_processed[column]).dt.dayofweek
            
            # Create polynomial features for highly correlated numeric columns
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = X_processed[numeric_cols].corr()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                # Create interaction features for highly correlated pairs
                for col1, col2 in high_corr_pairs[:3]:  # Limit to top 3 pairs
                    X_processed[f"{col1}_x_{col2}"] = X_processed[col1] * X_processed[col2]
            
            return X_processed
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return X

class DataValidator:
    """Validate preprocessed data quality"""
    
    def __init__(self):
        self.is_fitted = True
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'DataValidator':
        """Fit the data validator (no-op)"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data and return validation report"""
        try:
            validation_report = {
                "has_infinite": np.isinf(X.select_dtypes(include=[np.number])).sum().sum() > 0,
                "has_nan": X.isnull().sum().sum() > 0,
                "data_types": X.dtypes.value_counts().to_dict(),
                "shape": X.shape,
                "memory_usage": X.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Log validation results
            if validation_report["has_infinite"]:
                logger.warning("Data contains infinite values")
            if validation_report["has_nan"]:
                logger.warning("Data contains NaN values")
            
            return X
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return X
