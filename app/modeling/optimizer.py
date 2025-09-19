"""
Hyperparameter Optimizer - Hyperparameter optimization utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimization utilities"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.optimization_results = {}
    
    def grid_search(self, model: Any, param_grid: Dict[str, List], 
                   X: pd.DataFrame, y: pd.Series, 
                   scoring: str = "accuracy") -> Dict[str, Any]:
        """Perform grid search optimization"""
        try:
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state
            )
            
            grid_search.fit(X, y)
            
            results = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "best_estimator": grid_search.best_estimator_,
                "cv_results": grid_search.cv_results_,
                "optimization_method": "grid_search"
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            return {"error": str(e)}
    
    def random_search(self, model: Any, param_distributions: Dict[str, List], 
                     X: pd.DataFrame, y: pd.Series, 
                     n_iter: int = 100, scoring: str = "accuracy") -> Dict[str, Any]:
        """Perform random search optimization"""
        try:
            random_search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=n_iter,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state
            )
            
            random_search.fit(X, y)
            
            results = {
                "best_params": random_search.best_params_,
                "best_score": random_search.best_score_,
                "best_estimator": random_search.best_estimator_,
                "cv_results": random_search.cv_results_,
                "optimization_method": "random_search"
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in random search: {e}")
            return {"error": str(e)}
    
    def get_common_param_grids(self, model_type: str) -> Dict[str, List]:
        """Get common parameter grids for different model types"""
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0]
            },
            "svm": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["rbf", "linear", "poly"]
            },
            "logistic_regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        }
        
        return param_grids.get(model_type, {})
    
    def optimize_model(self, model: Any, model_type: str, 
                      X: pd.DataFrame, y: pd.Series,
                      method: str = "grid_search",
                      custom_params: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """Optimize model hyperparameters"""
        try:
            # Get parameter grid
            if custom_params:
                param_grid = custom_params
            else:
                param_grid = self.get_common_param_grids(model_type)
            
            if not param_grid:
                return {"error": f"No parameter grid found for {model_type}"}
            
            # Choose optimization method
            if method == "grid_search":
                results = self.grid_search(model, param_grid, X, y)
            elif method == "random_search":
                results = self.random_search(model, param_grid, X, y)
            else:
                return {"error": f"Unknown optimization method: {method}"}
            
            # Store results
            self.optimization_results[model_type] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return {"error": str(e)}
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of all optimization results"""
        try:
            summary_data = []
            
            for model_type, results in self.optimization_results.items():
                if "error" not in results:
                    summary_data.append({
                        "Model": model_type,
                        "Best Score": results.get("best_score", 0),
                        "Method": results.get("optimization_method", "unknown"),
                        "Best Params": str(results.get("best_params", {}))
                    })
            
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return pd.DataFrame()
