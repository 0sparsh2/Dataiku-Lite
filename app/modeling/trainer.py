"""
Model Trainer - Comprehensive model training with educational guidance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import time
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TrainingResult:
    """Results from model training"""
    model_name: str
    model: Any
    train_score: float
    test_score: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    learning_curve: Optional[Dict[str, List[float]]] = None

class ModelTrainer:
    """
    Comprehensive model trainer with educational guidance
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models: Dict[str, TrainingResult] = {}
        self.best_model: Optional[TrainingResult] = None
        self.training_logs = []
        
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series, 
                                  test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, TrainingResult]:
        """Train multiple classification models"""
        try:
            self._log("Starting classification model training")
            
            # Check if we can use stratified split
            min_samples_per_class = y.value_counts().min()
            can_stratify = min_samples_per_class >= 2 and len(y.unique()) > 1
            
            if can_stratify:
                # Use stratified split if possible
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state, stratify=y
                )
                self._log(f"Using stratified split (min samples per class: {min_samples_per_class})")
            else:
                # Use regular split if stratification not possible
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
                self._log(f"Using regular split (min samples per class: {min_samples_per_class})")
                
                # Check if we have enough data for classification
                if min_samples_per_class < 2:
                    raise ValueError(f"Insufficient data for classification. Minimum 2 samples per class required, but found {min_samples_per_class} samples in the smallest class. Consider using regression or clustering instead.")
            
            # Define models
            models = self._get_classification_models()
            
            # Train each model
            results = {}
            for name, model in models.items():
                self._log(f"Training {name}")
                result = self._train_single_model(
                    model, name, X_train, X_test, y_train, y_test, 
                    cv_folds, problem_type="classification"
                )
                results[name] = result
                self.trained_models[name] = result
            
            # Find best model
            self.best_model = max(results.values(), key=lambda x: x.cv_mean)
            self._log(f"Best model: {self.best_model.model_name} (CV Score: {self.best_model.cv_mean:.4f})")
            
            return results
            
        except Exception as e:
            self._log(f"Error in classification training: {str(e)}", level="ERROR")
            raise
    
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2, cv_folds: int = 5) -> Dict[str, TrainingResult]:
        """Train multiple regression models"""
        try:
            self._log("Starting regression model training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            # Define models
            models = self._get_regression_models()
            
            # Train each model
            results = {}
            for name, model in models.items():
                self._log(f"Training {name}")
                result = self._train_single_model(
                    model, name, X_train, X_test, y_train, y_test,
                    cv_folds, problem_type="regression"
                )
                results[name] = result
                self.trained_models[name] = result
            
            # Find best model
            self.best_model = min(results.values(), key=lambda x: x.cv_mean)  # Lower is better for regression
            self._log(f"Best model: {self.best_model.model_name} (CV Score: {self.best_model.cv_mean:.4f})")
            
            return results
            
        except Exception as e:
            self._log(f"Error in regression training: {str(e)}", level="ERROR")
            raise
    
    def train_clustering_models(self, X: pd.DataFrame, max_clusters: int = 10) -> Dict[str, TrainingResult]:
        """Train multiple clustering models"""
        try:
            self._log("Starting clustering model training")
            
            # Define models
            models = self._get_clustering_models(max_clusters)
            
            # Train each model
            results = {}
            for name, model in models.items():
                self._log(f"Training {name}")
                result = self._train_clustering_model(model, name, X)
                results[name] = result
                self.trained_models[name] = result
            
            return results
            
        except Exception as e:
            self._log(f"Error in clustering training: {str(e)}", level="ERROR")
            raise
    
    def _train_single_model(self, model: Any, name: str, X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                          cv_folds: int, problem_type: str) -> TrainingResult:
        """Train a single model with cross-validation"""
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Get probabilities for classification
        probabilities = None
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
        
        # Calculate scores
        if problem_type == "classification":
            train_score = accuracy_score(y_train, train_pred)
            test_score = accuracy_score(y_test, test_pred)
            cv_scorer = "accuracy"
        else:  # regression
            train_score = r2_score(y_train, train_pred)
            test_score = r2_score(y_test, test_pred)
            cv_scorer = "neg_mean_squared_error"
        
        # Cross-validation
        if problem_type == "classification":
            # Check if we can use stratified CV
            min_samples_per_class = y_train.value_counts().min()
            if min_samples_per_class >= 2 and len(y_train.unique()) > 1:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=cv_scorer)
        
        if problem_type == "regression":
            cv_scores = -cv_scores  # Convert back to positive MSE
        
        # Feature importance
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, "coef_"):
            feature_importance = dict(zip(X_train.columns, model.coef_))
        
        # Learning curve
        learning_curve = self._calculate_learning_curve(model, X_train, y_train, problem_type)
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model_name=name,
            model=model,
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            training_time=training_time,
            predictions=test_pred,
            probabilities=probabilities,
            feature_importance=feature_importance,
            learning_curve=learning_curve
        )
    
    def _train_clustering_model(self, model: Any, name: str, X: pd.DataFrame) -> TrainingResult:
        """Train a clustering model"""
        start_time = time.time()
        
        # Train model
        model.fit(X)
        
        # Get predictions
        predictions = model.predict(X)
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        try:
            silhouette = silhouette_score(X, predictions)
        except:
            silhouette = 0.0
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model_name=name,
            model=model,
            train_score=silhouette,
            test_score=silhouette,
            cv_scores=[silhouette],
            cv_mean=silhouette,
            cv_std=0.0,
            training_time=training_time,
            predictions=predictions
        )
    
    def _get_classification_models(self) -> Dict[str, Any]:
        """Get classification models"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        
        return {
            "Logistic Regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "SVM": SVC(random_state=self.random_state, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state)
        }
    
    def _get_regression_models(self) -> Dict[str, Any]:
        """Get regression models"""
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=self.random_state),
            "Lasso Regression": Lasso(random_state=self.random_state),
            "Elastic Net": ElasticNet(random_state=self.random_state),
            "Random Forest": RandomForestRegressor(random_state=self.random_state, n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(random_state=self.random_state),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=self.random_state)
        }
    
    def _get_clustering_models(self, max_clusters: int) -> Dict[str, Any]:
        """Get clustering models"""
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.mixture import GaussianMixture
        
        return {
            "K-Means": KMeans(n_clusters=3, random_state=self.random_state),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "Gaussian Mixture": GaussianMixture(n_components=3, random_state=self.random_state),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }
    
    def _calculate_learning_curve(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                problem_type: str) -> Dict[str, List[float]]:
        """Calculate learning curve for the model"""
        from sklearn.model_selection import learning_curve
        
        try:
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=3, random_state=self.random_state
            )
            
            return {
                "train_sizes": train_sizes_abs.tolist(),
                "train_scores": train_scores.mean(axis=1).tolist(),
                "val_scores": val_scores.mean(axis=1).tolist(),
                "train_std": train_scores.std(axis=1).tolist(),
                "val_std": val_scores.std(axis=1).tolist()
            }
        except Exception as e:
            self._log(f"Error calculating learning curve: {str(e)}", level="WARNING")
            return {}
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        if not self.trained_models:
            return pd.DataFrame()
        
        comparison_data = []
        for name, result in self.trained_models.items():
            comparison_data.append({
                "Model": name,
                "Train Score": result.train_score,
                "Test Score": result.test_score,
                "CV Mean": result.cv_mean,
                "CV Std": result.cv_std,
                "Training Time": result.training_time
            })
        
        return pd.DataFrame(comparison_data).sort_values("CV Mean", ascending=False)
    
    def get_best_model(self) -> Optional[TrainingResult]:
        """Get the best performing model"""
        return self.best_model
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            "model": self.trained_models[model_name].model,
            "model_name": model_name,
            "feature_names": list(self.trained_models[model_name].model.feature_names_in_) if hasattr(self.trained_models[model_name].model, 'feature_names_in_') else None
        }
        
        joblib.dump(model_data, filepath)
        self._log(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self._log(f"Model loaded from {filepath}")
        return model_data["model"]
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Add log entry"""
        log_entry = {
            "timestamp": pd.Timestamp.now(),
            "level": level,
            "message": message
        }
        self.training_logs.append(log_entry)
        
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_training_logs(self) -> List[Dict[str, Any]]:
        """Get training logs"""
        return self.training_logs
    
    def detect_problem_type(self, y: pd.Series) -> Dict[str, Any]:
        """Detect the appropriate problem type based on target variable"""
        analysis = {
            "is_numeric": pd.api.types.is_numeric_dtype(y),
            "unique_values": len(y.unique()),
            "min_samples_per_class": y.value_counts().min() if len(y.unique()) > 1 else 0,
            "suggested_type": None,
            "warnings": []
        }
        
        if analysis["is_numeric"]:
            if analysis["unique_values"] <= 10 and analysis["min_samples_per_class"] >= 2:
                analysis["suggested_type"] = "Classification"
            elif analysis["unique_values"] > 10:
                analysis["suggested_type"] = "Regression"
            else:
                analysis["suggested_type"] = "Clustering"
                analysis["warnings"].append("Insufficient samples per class for classification. Consider clustering instead.")
        else:
            if analysis["min_samples_per_class"] >= 2:
                analysis["suggested_type"] = "Classification"
            else:
                analysis["suggested_type"] = "Clustering"
                analysis["warnings"].append("Insufficient samples per class for classification. Consider clustering instead.")
        
        return analysis
