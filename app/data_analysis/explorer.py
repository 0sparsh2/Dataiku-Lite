"""
Data Explorer - Interactive data exploration utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataExplorer:
    """Interactive data exploration utilities"""
    
    def __init__(self):
        pass
    
    def explore_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic data exploration"""
        try:
            exploration_results = {
                "basic_info": self._get_basic_info(data),
                "data_types": self._analyze_data_types(data),
                "missing_patterns": self._analyze_missing_patterns(data),
                "numeric_summary": self._get_numeric_summary(data),
                "categorical_summary": self._get_categorical_summary(data)
            }
            
            return exploration_results
            
        except Exception as e:
            logger.error(f"Error in data exploration: {e}")
            return {"error": str(e)}
    
    def _get_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            "shape": data.shape,
            "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,
            "columns": list(data.columns),
            "index_type": str(data.index.dtype)
        }
    
    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types distribution"""
        return {
            "dtype_counts": data.dtypes.value_counts().to_dict(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist()
        }
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        missing_data = data.isnull()
        return {
            "total_missing": missing_data.sum().sum(),
            "missing_percentage": (missing_data.sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            "columns_with_missing": missing_data.sum()[missing_data.sum() > 0].to_dict(),
            "rows_with_missing": missing_data.any(axis=1).sum()
        }
    
    def _get_numeric_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for numeric columns"""
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {}
        
        return numeric_data.describe().to_dict()
    
    def _get_categorical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary for categorical columns"""
        categorical_data = data.select_dtypes(include=['object', 'category'])
        if categorical_data.empty:
            return {}
        
        summary = {}
        for col in categorical_data.columns:
            summary[col] = {
                "unique_count": categorical_data[col].nunique(),
                "most_common": categorical_data[col].value_counts().head(5).to_dict(),
                "missing_count": categorical_data[col].isnull().sum()
            }
        
        return summary
