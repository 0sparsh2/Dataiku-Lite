"""
Data Profiler - Comprehensive data profiling and analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ColumnProfile:
    """Profile for a single column"""
    name: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    is_numeric: bool
    is_categorical: bool
    is_datetime: bool
    is_text: bool
    statistics: Dict[str, Any] = None
    distribution_info: Dict[str, Any] = None
    quality_issues: List[str] = None

class DataProfiler:
    """
    Comprehensive data profiler that analyzes data characteristics
    and provides detailed insights for educational purposes
    """
    
    def __init__(self):
        self.profiles: Dict[str, ColumnProfile] = {}
        self.overall_stats: Dict[str, Any] = {}
        
    def profile_data(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive profile of the dataset
        """
        try:
            logger.info(f"Profiling dataset with shape {data.shape}")
            
            # Basic dataset info
            self.overall_stats = self._get_overall_stats(data)
            
            # Profile each column
            self.profiles = {}
            for column in data.columns:
                self.profiles[column] = self._profile_column(data[column], column)
            
            # Generate insights
            insights = self._generate_insights(data, target_column)
            
            # Quality assessment
            quality_report = self._generate_quality_report(data)
            
            return {
                "overall_stats": self.overall_stats,
                "column_profiles": {k: self._profile_to_dict(v) for k, v in self.profiles.items()},
                "insights": insights,
                "quality_report": quality_report,
                "recommendations": self._generate_recommendations(data, target_column)
            }
            
        except Exception as e:
            logger.error(f"Error profiling data: {e}")
            return {"error": str(e)}
    
    def _get_overall_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get overall dataset statistics"""
        return {
            "shape": data.shape,
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2,
            "total_cells": data.shape[0] * data.shape[1],
            "missing_cells": data.isnull().sum().sum(),
            "missing_percentage": (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            "duplicate_rows": data.duplicated().sum(),
            "duplicate_percentage": (data.duplicated().sum() / len(data)) * 100,
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(data.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(data.select_dtypes(include=['datetime64']).columns),
            "boolean_columns": len(data.select_dtypes(include=['bool']).columns)
        }
    
    def _profile_column(self, series: pd.Series, column_name: str) -> ColumnProfile:
        """Create detailed profile for a single column"""
        try:
            # Basic info
            null_count = series.isnull().sum()
            null_percentage = (null_count / len(series)) * 100
            unique_count = series.nunique()
            unique_percentage = (unique_count / len(series)) * 100
            
            # Data type classification
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_categorical = pd.api.types.is_categorical_dtype(series) or series.dtype == 'object'
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            is_text = series.dtype == 'object' and not is_categorical
            
            # Initialize profile
            profile = ColumnProfile(
                name=column_name,
                dtype=str(series.dtype),
                null_count=null_count,
                null_percentage=null_percentage,
                unique_count=unique_count,
                unique_percentage=unique_percentage,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                is_datetime=is_datetime,
                is_text=is_text,
                quality_issues=[]
            )
            
            # Add statistics based on data type
            if is_numeric:
                profile.statistics = self._get_numeric_statistics(series)
                profile.distribution_info = self._analyze_distribution(series)
            elif is_categorical:
                profile.statistics = self._get_categorical_statistics(series)
            elif is_datetime:
                profile.statistics = self._get_datetime_statistics(series)
            elif is_text:
                profile.statistics = self._get_text_statistics(series)
            
            # Identify quality issues
            profile.quality_issues = self._identify_quality_issues(series, profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling column {column_name}: {e}")
            return ColumnProfile(
                name=column_name,
                dtype=str(series.dtype),
                null_count=0,
                null_percentage=0,
                unique_count=0,
                unique_percentage=0,
                is_numeric=False,
                is_categorical=False,
                is_datetime=False,
                is_text=False,
                quality_issues=[f"Error profiling: {str(e)}"]
            )
    
    def _get_numeric_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for numeric columns"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            stats_dict = {
                "count": series.count(),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "median": series.median(),
                "q25": series.quantile(0.25),
                "q75": series.quantile(0.75),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "zeros": (series == 0).sum(),
                "negatives": (series < 0).sum(),
                "positives": (series > 0).sum()
            }
            
            # Add percentiles
            for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                stats_dict[f"p{p}"] = series.quantile(p/100)
            
            return stats_dict
    
    def _get_categorical_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for categorical columns"""
        value_counts = series.value_counts()
        
        return {
            "count": series.count(),
            "unique_values": series.nunique(),
            "most_common": value_counts.head(5).to_dict(),
            "least_common": value_counts.tail(5).to_dict(),
            "entropy": self._calculate_entropy(series),
            "mode": series.mode().iloc[0] if not series.mode().empty else None,
            "mode_frequency": value_counts.iloc[0] if not value_counts.empty else 0
        }
    
    def _get_datetime_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for datetime columns"""
        try:
            return {
                "count": series.count(),
                "min_date": series.min(),
                "max_date": series.max(),
                "date_range_days": (series.max() - series.min()).days,
                "unique_dates": series.nunique(),
                "has_time_component": any(series.dropna().dt.time != pd.Timestamp('00:00:00').time()),
                "day_of_week_dist": series.dt.day_name().value_counts().to_dict(),
                "month_dist": series.dt.month_name().value_counts().to_dict(),
                "year_dist": series.dt.year.value_counts().to_dict()
            }
        except Exception as e:
            return {"error": f"Error analyzing datetime: {str(e)}"}
    
    def _get_text_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Get statistics for text columns"""
        text_series = series.dropna().astype(str)
        
        return {
            "count": len(text_series),
            "avg_length": text_series.str.len().mean(),
            "min_length": text_series.str.len().min(),
            "max_length": text_series.str.len().max(),
            "empty_strings": (text_series == "").sum(),
            "whitespace_only": text_series.str.strip().eq("").sum(),
            "contains_numbers": text_series.str.contains(r'\d').sum(),
            "contains_special_chars": text_series.str.contains(r'[^a-zA-Z0-9\s]').sum()
        }
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze distribution characteristics"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Remove nulls for analysis
            clean_series = series.dropna()
            
            if len(clean_series) < 3:
                return {"error": "Insufficient data for distribution analysis"}
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(clean_series) if len(clean_series) <= 5000 else (None, None)
            ks_stat, ks_p = stats.kstest(clean_series, 'norm', args=(clean_series.mean(), clean_series.std()))
            
            return {
                "is_normal": shapiro_p > 0.05 if shapiro_p else ks_p > 0.05,
                "shapiro_stat": shapiro_stat,
                "shapiro_p": shapiro_p,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "skewness": clean_series.skew(),
                "kurtosis": clean_series.kurtosis(),
                "distribution_type": self._classify_distribution(clean_series)
            }
    
    def _classify_distribution(self, series: pd.Series) -> str:
        """Classify the type of distribution"""
        skewness = abs(series.skew())
        kurtosis = series.kurtosis()
        
        if skewness < 0.5:
            if kurtosis < 3:
                return "approximately normal"
            else:
                return "normal with heavy tails"
        elif skewness < 1:
            return "moderately skewed"
        else:
            return "highly skewed"
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy for categorical data"""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    def _identify_quality_issues(self, series: pd.Series, profile: ColumnProfile) -> List[str]:
        """Identify data quality issues"""
        issues = []
        
        # Missing values
        if profile.null_percentage > 50:
            issues.append("High missing value percentage (>50%)")
        elif profile.null_percentage > 20:
            issues.append("Moderate missing value percentage (>20%)")
        
        # Uniqueness issues
        if profile.unique_percentage < 1:
            issues.append("Very low uniqueness (<1%)")
        elif profile.unique_percentage < 5:
            issues.append("Low uniqueness (<5%)")
        
        # Numeric-specific issues
        if profile.is_numeric and profile.statistics:
            stats = profile.statistics
            if stats.get('zeros', 0) > len(series) * 0.8:
                issues.append("High percentage of zeros (>80%)")
            if stats.get('skewness', 0) > 2:
                issues.append("Highly skewed distribution")
            if stats.get('kurtosis', 0) > 3:
                issues.append("Heavy-tailed distribution")
        
        # Categorical-specific issues
        if profile.is_categorical and profile.statistics:
            stats = profile.statistics
            if stats.get('mode_frequency', 0) > len(series) * 0.9:
                issues.append("Highly imbalanced categories (>90% in one category)")
            if stats.get('entropy', 0) < 1:
                issues.append("Low entropy (low diversity)")
        
        return issues
    
    def _generate_insights(self, data: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Generate high-level insights about the dataset"""
        insights = {
            "data_quality_score": self._calculate_quality_score(),
            "key_findings": [],
            "correlation_insights": [],
            "distribution_insights": []
        }
        
        # Data quality score
        quality_score = self._calculate_quality_score()
        insights["data_quality_score"] = quality_score
        
        # Key findings
        if self.overall_stats["missing_percentage"] > 20:
            insights["key_findings"].append("Dataset has significant missing values that need attention")
        
        if self.overall_stats["duplicate_percentage"] > 10:
            insights["key_findings"].append("Dataset contains duplicate rows that should be reviewed")
        
        # Correlation insights for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if high_corr_pairs:
                insights["correlation_insights"] = high_corr_pairs[:5]  # Top 5
        
        return insights
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Deduct for missing values
        missing_penalty = min(self.overall_stats["missing_percentage"] * 2, 40)
        score -= missing_penalty
        
        # Deduct for duplicates
        duplicate_penalty = min(self.overall_stats["duplicate_percentage"] * 2, 20)
        score -= duplicate_penalty
        
        # Deduct for quality issues
        total_issues = sum(len(profile.quality_issues) for profile in self.profiles.values())
        issue_penalty = min(total_issues * 2, 30)
        score -= issue_penalty
        
        return max(score, 0)
    
    def _generate_recommendations(self, data: pd.DataFrame, target_column: Optional[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Missing value recommendations
        if self.overall_stats["missing_percentage"] > 5:
            recommendations.append("Consider strategies for handling missing values (imputation, deletion, or modeling)")
        
        # Duplicate recommendations
        if self.overall_stats["duplicate_percentage"] > 5:
            recommendations.append("Review and potentially remove duplicate rows")
        
        # Categorical encoding recommendations
        categorical_cols = [col for col, profile in self.profiles.items() if profile.is_categorical]
        if categorical_cols:
            recommendations.append(f"Encode {len(categorical_cols)} categorical variables for modeling")
        
        # Feature selection recommendations
        if len(data.columns) > 20:
            recommendations.append("Consider feature selection to reduce dimensionality")
        
        # Scaling recommendations
        numeric_cols = [col for col, profile in self.profiles.items() if profile.is_numeric]
        if numeric_cols:
            recommendations.append("Scale numeric features for algorithms sensitive to scale")
        
        return recommendations
    
    def _generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report"""
        try:
            quality_issues = []
            
            # Check for missing values
            missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            if missing_percentage > 20:
                quality_issues.append(f"High missing value percentage: {missing_percentage:.1f}%")
            
            # Check for duplicates
            duplicate_percentage = (data.duplicated().sum() / len(data)) * 100
            if duplicate_percentage > 10:
                quality_issues.append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
            
            # Check for outliers in numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(data[col].dropna()) > 10:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers > len(data) * 0.1:  # More than 10% outliers
                        quality_issues.append(f"High outlier percentage in {col}")
            
            return {
                "quality_issues": quality_issues,
                "missing_percentage": missing_percentage,
                "duplicate_percentage": duplicate_percentage,
                "overall_quality_score": max(0, 100 - len(quality_issues) * 20)
            }
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {"quality_issues": [], "overall_quality_score": 0}
    
    def _profile_to_dict(self, profile: ColumnProfile) -> Dict[str, Any]:
        """Convert ColumnProfile to dictionary for JSON serialization"""
        return {
            "name": profile.name,
            "dtype": profile.dtype,
            "null_count": profile.null_count,
            "null_percentage": profile.null_percentage,
            "unique_count": profile.unique_count,
            "unique_percentage": profile.unique_percentage,
            "is_numeric": profile.is_numeric,
            "is_categorical": profile.is_categorical,
            "is_datetime": profile.is_datetime,
            "is_text": profile.is_text,
            "statistics": profile.statistics,
            "distribution_info": profile.distribution_info,
            "quality_issues": profile.quality_issues
        }
