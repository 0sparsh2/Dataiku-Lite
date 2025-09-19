"""
Data Validators - Data quality validation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    message: str
    details: Dict[str, Any] = None
    severity: str = "warning"  # info, warning, error

class DataValidator:
    """Comprehensive data validation"""
    
    def __init__(self, rules: Optional[Dict[str, bool]] = None):
        self.rules = rules or {
            "check_infinite": True,
            "check_nan": True,
            "check_duplicates": True,
            "check_data_types": True,
            "check_value_ranges": True,
            "check_outliers": True,
            "check_correlations": True
        }
        self.is_fitted = True
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> 'DataValidator':
        """Fit the data validator (no-op)"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data and return validation report"""
        try:
            validation_results = self.validate_data(X)
            
            # Log results
            for result in validation_results:
                if not result.passed:
                    if result.severity == "error":
                        logger.error(f"Validation error: {result.message}")
                    else:
                        logger.warning(f"Validation warning: {result.message}")
                else:
                    logger.info(f"Validation passed: {result.message}")
            
            return X
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return X
    
    def validate_data(self, X: pd.DataFrame) -> List[ValidationResult]:
        """Perform comprehensive data validation"""
        results = []
        
        # Check for infinite values
        if self.rules.get("check_infinite", True):
            results.append(self._check_infinite_values(X))
        
        # Check for NaN values
        if self.rules.get("check_nan", True):
            results.append(self._check_nan_values(X))
        
        # Check for duplicates
        if self.rules.get("check_duplicates", True):
            results.append(self._check_duplicates(X))
        
        # Check data types
        if self.rules.get("check_data_types", True):
            results.append(self._check_data_types(X))
        
        # Check value ranges
        if self.rules.get("check_value_ranges", True):
            results.append(self._check_value_ranges(X))
        
        # Check for outliers
        if self.rules.get("check_outliers", True):
            results.append(self._check_outliers(X))
        
        # Check correlations
        if self.rules.get("check_correlations", True):
            results.append(self._check_correlations(X))
        
        return results
    
    def _check_infinite_values(self, X: pd.DataFrame) -> ValidationResult:
        """Check for infinite values"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        infinite_count = np.isinf(X[numeric_cols]).sum().sum()
        
        if infinite_count > 0:
            infinite_cols = X[numeric_cols].columns[np.isinf(X[numeric_cols]).any()].tolist()
            return ValidationResult(
                passed=False,
                message=f"Found {infinite_count} infinite values in columns: {infinite_cols}",
                details={"infinite_count": infinite_count, "columns": infinite_cols},
                severity="error"
            )
        else:
            return ValidationResult(
                passed=True,
                message="No infinite values found",
                details={"infinite_count": 0}
            )
    
    def _check_nan_values(self, X: pd.DataFrame) -> ValidationResult:
        """Check for NaN values"""
        nan_count = X.isnull().sum().sum()
        nan_percentage = (nan_count / (X.shape[0] * X.shape[1])) * 100
        
        if nan_count > 0:
            nan_cols = X.columns[X.isnull().any()].tolist()
            return ValidationResult(
                passed=False,
                message=f"Found {nan_count} NaN values ({nan_percentage:.2f}%) in columns: {nan_cols}",
                details={"nan_count": nan_count, "nan_percentage": nan_percentage, "columns": nan_cols},
                severity="warning" if nan_percentage < 10 else "error"
            )
        else:
            return ValidationResult(
                passed=True,
                message="No NaN values found",
                details={"nan_count": 0, "nan_percentage": 0}
            )
    
    def _check_duplicates(self, X: pd.DataFrame) -> ValidationResult:
        """Check for duplicate rows"""
        duplicate_count = X.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(X)) * 100
        
        if duplicate_count > 0:
            return ValidationResult(
                passed=False,
                message=f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}%)",
                details={"duplicate_count": duplicate_count, "duplicate_percentage": duplicate_percentage},
                severity="warning" if duplicate_percentage < 5 else "error"
            )
        else:
            return ValidationResult(
                passed=True,
                message="No duplicate rows found",
                details={"duplicate_count": 0, "duplicate_percentage": 0}
            )
    
    def _check_data_types(self, X: pd.DataFrame) -> ValidationResult:
        """Check for data type consistency"""
        mixed_type_cols = []
        inconsistent_cols = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                # Check if column has mixed types
                try:
                    pd.to_numeric(X[col], errors='raise')
                except:
                    mixed_type_cols.append(col)
            
            # Check for inconsistent data types within columns
            if X[col].dtype == 'object':
                try:
                    unique_types = X[col].apply(type).nunique()
                    if unique_types > 1:
                        inconsistent_cols.append(col)
                except Exception:
                    # Skip if there's an error checking types
                    pass
        
        if mixed_type_cols or inconsistent_cols:
            return ValidationResult(
                passed=False,
                message=f"Found data type issues in columns: mixed_types={mixed_type_cols}, inconsistent={inconsistent_cols}",
                details={"mixed_type_cols": mixed_type_cols, "inconsistent_cols": inconsistent_cols},
                severity="warning"
            )
        else:
            return ValidationResult(
                passed=True,
                message="All columns have consistent data types",
                details={"mixed_type_cols": [], "inconsistent_cols": []}
            )
    
    def _check_value_ranges(self, X: pd.DataFrame) -> ValidationResult:
        """Check for extreme values in numeric columns"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        extreme_value_cols = []
        
        for col in numeric_cols:
            if X[col].min() < -1e10 or X[col].max() > 1e10:
                extreme_value_cols.append(col)
        
        if extreme_value_cols:
            return ValidationResult(
                passed=False,
                message=f"Found extreme values in columns: {extreme_value_cols}",
                details={"extreme_value_cols": extreme_value_cols},
                severity="warning"
            )
        else:
            return ValidationResult(
                passed=True,
                message="All numeric columns have reasonable value ranges",
                details={"extreme_value_cols": []}
            )
    
    def _check_outliers(self, X: pd.DataFrame) -> ValidationResult:
        """Check for outliers in numeric columns"""
        from scipy import stats
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        
        for col in numeric_cols:
            if len(X[col].dropna()) > 10:  # Need sufficient data for outlier detection
                z_scores = np.abs(stats.zscore(X[col].dropna()))
                outlier_count = (z_scores > 3).sum()
                outlier_percentage = (outlier_count / len(X[col].dropna())) * 100
                
                if outlier_percentage > 5:  # More than 5% outliers
                    outlier_cols.append({
                        "column": col,
                        "outlier_count": outlier_count,
                        "outlier_percentage": outlier_percentage
                    })
        
        if outlier_cols:
            return ValidationResult(
                passed=False,
                message=f"Found significant outliers in {len(outlier_cols)} columns",
                details={"outlier_cols": outlier_cols},
                severity="warning"
            )
        else:
            return ValidationResult(
                passed=True,
                message="No significant outliers found",
                details={"outlier_cols": []}
            )
    
    def _check_correlations(self, X: pd.DataFrame) -> ValidationResult:
        """Check for high correlations between features"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return ValidationResult(
                passed=True,
                message="Insufficient numeric columns for correlation analysis",
                details={"high_corr_pairs": []}
            )
        
        corr_matrix = X[numeric_cols].corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:  # Very high correlation
                    high_corr_pairs.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": corr_val
                    })
        
        if high_corr_pairs:
            return ValidationResult(
                passed=False,
                message=f"Found {len(high_corr_pairs)} highly correlated feature pairs",
                details={"high_corr_pairs": high_corr_pairs},
                severity="warning"
            )
        else:
            return ValidationResult(
                passed=True,
                message="No highly correlated features found",
                details={"high_corr_pairs": []}
            )
    
    def get_validation_summary(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of all validation results"""
        results = self.validate_data(X)
        
        summary = {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r.passed),
            "failed_checks": sum(1 for r in results if not r.passed),
            "warnings": sum(1 for r in results if not r.passed and r.severity == "warning"),
            "errors": sum(1 for r in results if not r.passed and r.severity == "error"),
            "results": results
        }
        
        return summary
