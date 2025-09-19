"""
Data Quality Checker - Data quality assessment utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Data quality assessment utilities"""
    
    def __init__(self):
        pass
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks"""
        try:
            quality_results = {
                "completeness": self._check_completeness(data),
                "consistency": self._check_consistency(data),
                "accuracy": self._check_accuracy(data),
                "validity": self._check_validity(data),
                "uniqueness": self._check_uniqueness(data),
                "overall_score": 0
            }
            
            # Calculate overall quality score
            scores = [result.get("score", 0) for result in quality_results.values() if isinstance(result, dict) and "score" in result]
            quality_results["overall_score"] = np.mean(scores) if scores else 0
            
            return quality_results
            
        except Exception as e:
            logger.error(f"Error in data quality checking: {e}")
            return {"error": str(e)}
    
    def _check_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        missing_data = data.isnull()
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = missing_data.sum().sum()
        
        completeness_score = max(0, 100 - (missing_cells / total_cells) * 100)
        
        return {
            "score": completeness_score,
            "missing_cells": missing_cells,
            "missing_percentage": (missing_cells / total_cells) * 100,
            "complete_rows": (~missing_data.any(axis=1)).sum(),
            "complete_columns": (~missing_data.any(axis=0)).sum()
        }
    
    def _check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        consistency_issues = []
        
        # Check for mixed data types in columns
        for col in data.columns:
            if data[col].dtype == 'object':
                unique_types = data[col].apply(type).nunique()
                if unique_types > 1:
                    consistency_issues.append(f"Mixed types in {col}")
        
        # Check for inconsistent formats
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for inconsistent string lengths (potential formatting issues)
                string_lengths = data[col].astype(str).str.len()
                if string_lengths.std() > string_lengths.mean():
                    consistency_issues.append(f"Inconsistent string lengths in {col}")
        
        consistency_score = max(0, 100 - len(consistency_issues) * 10)
        
        return {
            "score": consistency_score,
            "issues": consistency_issues,
            "issue_count": len(consistency_issues)
        }
    
    def _check_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy"""
        accuracy_issues = []
        
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(data[col].dropna()) > 10:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(data) * 0.05:  # More than 5% outliers
                    accuracy_issues.append(f"High outlier percentage in {col}")
        
        # Check for impossible values
        for col in numeric_cols:
            if data[col].min() < -1e10 or data[col].max() > 1e10:
                accuracy_issues.append(f"Extreme values in {col}")
        
        accuracy_score = max(0, 100 - len(accuracy_issues) * 15)
        
        return {
            "score": accuracy_score,
            "issues": accuracy_issues,
            "issue_count": len(accuracy_issues)
        }
    
    def _check_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data validity"""
        validity_issues = []
        
        # Check for invalid values
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check for empty strings
                empty_strings = (data[col] == '').sum()
                if empty_strings > 0:
                    validity_issues.append(f"Empty strings in {col}")
                
                # Check for whitespace-only strings
                whitespace_only = data[col].astype(str).str.strip().eq('').sum()
                if whitespace_only > 0:
                    validity_issues.append(f"Whitespace-only strings in {col}")
        
        validity_score = max(0, 100 - len(validity_issues) * 10)
        
        return {
            "score": validity_score,
            "issues": validity_issues,
            "issue_count": len(validity_issues)
        }
    
    def _check_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data uniqueness"""
        duplicate_rows = data.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(data)) * 100
        
        # Check for potential key columns (high uniqueness)
        potential_keys = []
        for col in data.columns:
            uniqueness = data[col].nunique() / len(data)
            if uniqueness > 0.95:  # More than 95% unique
                potential_keys.append(col)
        
        uniqueness_score = max(0, 100 - duplicate_percentage)
        
        return {
            "score": uniqueness_score,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "potential_keys": potential_keys
        }
