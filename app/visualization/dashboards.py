"""
Dashboard Builder - Interactive dashboard creation utilities
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DashboardBuilder:
    """Interactive dashboard builder"""
    
    def __init__(self):
        self.default_colors = px.colors.qualitative.Set3
    
    def create_data_overview_dashboard(self, data: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create comprehensive data overview dashboard"""
        try:
            dashboard = {}
            
            # Basic statistics
            dashboard["shape_info"] = self._create_shape_info_chart(data)
            
            # Missing values
            if data.isnull().sum().sum() > 0:
                dashboard["missing_values"] = self._create_missing_values_chart(data)
            
            # Data types distribution
            dashboard["data_types"] = self._create_data_types_chart(data)
            
            # Numeric columns summary
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                dashboard["numeric_summary"] = self._create_numeric_summary_chart(data, numeric_cols)
            
            # Categorical columns summary
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                dashboard["categorical_summary"] = self._create_categorical_summary_chart(data, categorical_cols)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating data overview dashboard: {e}")
            return {}
    
    def create_model_performance_dashboard(self, model_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create model performance dashboard"""
        try:
            dashboard = {}
            
            # Model comparison
            if "comparison_data" in model_results:
                dashboard["model_comparison"] = self._create_model_comparison_chart(model_results["comparison_data"])
            
            # Learning curves
            if "learning_curves" in model_results:
                dashboard["learning_curves"] = self._create_learning_curves_chart(model_results["learning_curves"])
            
            # Feature importance
            if "feature_importance" in model_results:
                dashboard["feature_importance"] = self._create_feature_importance_chart(model_results["feature_importance"])
            
            # Confusion matrix
            if "confusion_matrix" in model_results:
                dashboard["confusion_matrix"] = self._create_confusion_matrix_chart(model_results["confusion_matrix"])
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating model performance dashboard: {e}")
            return {}
    
    def _create_shape_info_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create shape information chart"""
        fig = go.Figure()
        
        # Add text annotations
        fig.add_annotation(
            x=0.5, y=0.7,
            text=f"Rows: {data.shape[0]:,}",
            showarrow=False,
            font=dict(size=20, color="blue")
        )
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Columns: {data.shape[1]}",
            showarrow=False,
            font=dict(size=20, color="green")
        )
        
        fig.add_annotation(
            x=0.5, y=0.3,
            text=f"Memory: {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            showarrow=False,
            font=dict(size=20, color="orange")
        )
        
        fig.update_layout(
            title="Dataset Overview",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300
        )
        
        return fig
    
    def _create_missing_values_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create missing values chart"""
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        fig = go.Figure(data=go.Bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Missing Count",
            yaxis_title="Columns",
            height=max(400, len(missing_data) * 30)
        )
        
        return fig
    
    def _create_data_types_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create data types distribution chart"""
        dtype_counts = data.dtypes.value_counts()
        
        fig = go.Figure(data=go.Pie(
            labels=dtype_counts.index,
            values=dtype_counts.values,
            hole=0.3
        ))
        
        fig.update_layout(
            title="Data Types Distribution",
            height=400
        )
        
        return fig
    
    def _create_numeric_summary_chart(self, data: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Create numeric columns summary chart"""
        summary_stats = data[numeric_cols].describe()
        
        fig = go.Figure()
        
        for stat in ['mean', 'std', 'min', 'max']:
            if stat in summary_stats.index:
                fig.add_trace(go.Bar(
                    name=stat.title(),
                    x=numeric_cols,
                    y=summary_stats.loc[stat],
                    text=summary_stats.loc[stat].round(2),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title="Numeric Columns Summary",
            xaxis_title="Columns",
            yaxis_title="Value",
            barmode='group',
            height=400
        )
        
        return fig
    
    def _create_categorical_summary_chart(self, data: pd.DataFrame, categorical_cols: List[str]) -> go.Figure:
        """Create categorical columns summary chart"""
        unique_counts = [data[col].nunique() for col in categorical_cols]
        
        fig = go.Figure(data=go.Bar(
            x=categorical_cols,
            y=unique_counts,
            marker_color='lightblue',
            text=unique_counts,
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Categorical Columns - Unique Values Count",
            xaxis_title="Columns",
            yaxis_title="Unique Values",
            height=400
        )
        
        return fig
    
    def _create_model_comparison_chart(self, comparison_data: pd.DataFrame) -> go.Figure:
        """Create model comparison chart"""
        fig = go.Figure(data=go.Bar(
            x=comparison_data['Model'],
            y=comparison_data['CV Mean'],
            error_y=dict(type='data', array=comparison_data.get('CV Std', [0] * len(comparison_data))),
            marker_color=self.default_colors[:len(comparison_data)]
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="CV Score",
            height=400
        )
        
        return fig
    
    def _create_learning_curves_chart(self, learning_curves: Dict[str, Any]) -> go.Figure:
        """Create learning curves chart"""
        fig = go.Figure()
        
        for model_name, curve_data in learning_curves.items():
            fig.add_trace(go.Scatter(
                x=curve_data['train_sizes'],
                y=curve_data['train_scores'],
                mode='lines+markers',
                name=f'{model_name} - Training',
                line=dict(dash='solid')
            ))
            
            fig.add_trace(go.Scatter(
                x=curve_data['train_sizes'],
                y=curve_data['val_scores'],
                mode='lines+markers',
                name=f'{model_name} - Validation',
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
            height=400
        )
        
        return fig
    
    def _create_feature_importance_chart(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create feature importance chart"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        fig = go.Figure(data=go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(features) * 30)
        )
        
        return fig
    
    def _create_confusion_matrix_chart(self, confusion_matrix: np.ndarray) -> go.Figure:
        """Create confusion matrix chart"""
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        return fig
