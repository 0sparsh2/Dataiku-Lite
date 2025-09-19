"""
Data Visualization Charts - Interactive plotting utilities
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Interactive data visualization utilities"""
    
    def __init__(self):
        self.default_colors = px.colors.qualitative.Set3
    
    def create_missing_values_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create heatmap showing missing values pattern"""
        try:
            missing_data = data.isnull()
            
            fig = go.Figure(data=go.Heatmap(
                z=missing_data.T.astype(int),
                x=list(range(len(data))),
                y=list(data.columns),
                colorscale='Reds',
                showscale=True,
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Missing Values Pattern",
                xaxis_title="Row Index",
                yaxis_title="Columns",
                height=max(400, len(data.columns) * 30)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating missing values heatmap: {e}")
            return go.Figure()
    
    def create_correlation_heatmap(self, data: pd.DataFrame, method: str = 'pearson') -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return go.Figure()
            
            corr_matrix = numeric_data.corr(method=method)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f"Correlation Matrix ({method.title()})",
                height=max(400, len(corr_matrix.columns) * 40),
                width=max(400, len(corr_matrix.columns) * 40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return go.Figure()
    
    def create_distribution_plots(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
        """Create distribution plots for numeric columns"""
        try:
            if columns is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
            else:
                numeric_cols = [col for col in columns if col in data.columns and data[col].dtype in ['int64', 'float64']]
            
            if len(numeric_cols) == 0:
                return go.Figure()
            
            # Calculate subplot layout
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=numeric_cols,
                specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
            )
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1
                
                # Create histogram
                fig.add_trace(
                    go.Histogram(
                        x=data[col].dropna(),
                        name=col,
                        showlegend=False,
                        nbinsx=30
                    ),
                    row=row,
                    col=col_idx
                )
            
            fig.update_layout(
                title="Distribution of Numeric Columns",
                height=n_rows * 300,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {e}")
            return go.Figure()
    
    def create_box_plots(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
        """Create box plots for numeric columns"""
        try:
            if columns is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
            else:
                numeric_cols = [col for col in columns if col in data.columns and data[col].dtype in ['int64', 'float64']]
            
            if len(numeric_cols) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            for col in numeric_cols:
                fig.add_trace(go.Box(
                    y=data[col].dropna(),
                    name=col,
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title="Box Plots for Numeric Columns",
                yaxis_title="Value",
                height=max(400, len(numeric_cols) * 100)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating box plots: {e}")
            return go.Figure()
    
    def create_categorical_plots(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> go.Figure:
        """Create bar plots for categorical columns"""
        try:
            if columns is None:
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            else:
                categorical_cols = [col for col in columns if col in data.columns and data[col].dtype in ['object', 'category']]
            
            if len(categorical_cols) == 0:
                return go.Figure()
            
            # Calculate subplot layout
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=categorical_cols,
                specs=[[{"secondary_y": False}] * n_cols for _ in range(n_rows)]
            )
            
            for i, col in enumerate(categorical_cols):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1
                
                value_counts = data[col].value_counts().head(10)  # Top 10 categories
                
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        name=col,
                        showlegend=False
                    ),
                    row=row,
                    col=col_idx
                )
            
            fig.update_layout(
                title="Distribution of Categorical Columns",
                height=n_rows * 300,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating categorical plots: {e}")
            return go.Figure()
    
    def create_feature_importance_plot(self, feature_importance: Dict[str, float], 
                                     title: str = "Feature Importance") -> go.Figure:
        """Create feature importance bar plot"""
        try:
            if not feature_importance:
                return go.Figure()
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features)
            
            fig = go.Figure(data=go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color=self.default_colors[0]
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance",
                yaxis_title="Features",
                height=max(400, len(features) * 30)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return go.Figure()
    
    def create_learning_curve_plot(self, learning_curve: Dict[str, List[float]], 
                                 title: str = "Learning Curve") -> go.Figure:
        """Create learning curve plot"""
        try:
            if not learning_curve or 'train_sizes' not in learning_curve:
                return go.Figure()
            
            fig = go.Figure()
            
            # Training scores
            fig.add_trace(go.Scatter(
                x=learning_curve['train_sizes'],
                y=learning_curve['train_scores'],
                mode='lines+markers',
                name='Training Score',
                line=dict(color=self.default_colors[0])
            ))
            
            # Validation scores
            fig.add_trace(go.Scatter(
                x=learning_curve['train_sizes'],
                y=learning_curve['val_scores'],
                mode='lines+markers',
                name='Validation Score',
                line=dict(color=self.default_colors[1])
            ))
            
            # Add error bars if available
            if 'train_std' in learning_curve and 'val_std' in learning_curve:
                fig.add_trace(go.Scatter(
                    x=learning_curve['train_sizes'],
                    y=[a + b for a, b in zip(learning_curve['train_scores'], learning_curve['train_std'])],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=learning_curve['train_sizes'],
                    y=[a - b for a, b in zip(learning_curve['train_scores'], learning_curve['train_std'])],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({self.default_colors[0][1:3]}, 0.2)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Training Set Size",
                yaxis_title="Score",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating learning curve plot: {e}")
            return go.Figure()
    
    def create_model_comparison_plot(self, comparison_data: pd.DataFrame, 
                                   metric: str = "CV Mean") -> go.Figure:
        """Create model comparison bar plot"""
        try:
            if comparison_data.empty:
                return go.Figure()
            
            fig = go.Figure(data=go.Bar(
                x=comparison_data['Model'],
                y=comparison_data[metric],
                error_y=dict(type='data', array=comparison_data.get('CV Std', [0] * len(comparison_data))),
                marker_color=self.default_colors[:len(comparison_data)]
            ))
            
            fig.update_layout(
                title=f"Model Comparison - {metric}",
                xaxis_title="Models",
                yaxis_title=metric,
                height=400,
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {e}")
            return go.Figure()
    
    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray, 
                                   class_names: Optional[List[str]] = None) -> go.Figure:
        """Create confusion matrix heatmap"""
        try:
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
            
            fig = go.Figure(data=go.Heatmap(
                z=confusion_matrix,
                x=class_names,
                y=class_names,
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
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {e}")
            return go.Figure()
    
    def create_roc_curve_plot(self, fpr: List[float], tpr: List[float], 
                             auc_score: float, title: str = "ROC Curve") -> go.Figure:
        """Create ROC curve plot"""
        try:
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc_score:.3f})',
                line=dict(color=self.default_colors[0])
            ))
            
            # Diagonal line (random classifier)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating ROC curve plot: {e}")
            return go.Figure()
