"""
Model Registry - Model management and tracking utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import joblib
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Model registry for tracking and managing trained models"""
    
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.models = {}
        self.metadata_file = self.registry_path / "metadata.json"
        self._load_metadata()
    
    def register_model(self, model_name: str, model: Any, 
                      metadata: Dict[str, Any]) -> bool:
        """Register a new model in the registry"""
        try:
            # Create model directory
            model_dir = self.registry_path / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_file = model_dir / "model.pkl"
            joblib.dump(model, model_file)
            
            # Save metadata
            metadata["model_name"] = model_name
            metadata["created_at"] = datetime.now().isoformat()
            metadata["model_file"] = str(model_file)
            
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry
            self.models[model_name] = metadata
            self._save_metadata()
            
            logger.info(f"Model {model_name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load a model from the registry"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found in registry")
                return None
            
            model_file = self.models[model_name]["model_file"]
            model = joblib.load(model_file)
            
            logger.info(f"Model {model_name} loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model"""
        return self.models.get(model_name)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return [
            {
                "model_name": name,
                "created_at": metadata.get("created_at", "unknown"),
                "model_type": metadata.get("model_type", "unknown"),
                "performance": metadata.get("performance", {})
            }
            for name, metadata in self.models.items()
        ]
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from the registry"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found in registry")
                return False
            
            # Remove model directory
            model_dir = self.registry_path / model_name
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            
            # Remove from registry
            del self.models[model_name]
            self._save_metadata()
            
            logger.info(f"Model {model_name} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def compare_models(self, metric: str = "accuracy") -> pd.DataFrame:
        """Compare all models by a specific metric"""
        try:
            comparison_data = []
            
            for model_name, metadata in self.models.items():
                performance = metadata.get("performance", {})
                if metric in performance:
                    comparison_data.append({
                        "Model": model_name,
                        "Metric": performance[metric],
                        "Model Type": metadata.get("model_type", "unknown"),
                        "Created At": metadata.get("created_at", "unknown")
                    })
            
            if not comparison_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(comparison_data)
            return df.sort_values("Metric", ascending=False)
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return pd.DataFrame()
    
    def get_best_model(self, metric: str = "accuracy") -> Optional[str]:
        """Get the best model based on a metric"""
        try:
            comparison_df = self.compare_models(metric)
            if comparison_df.empty:
                return None
            
            return comparison_df.iloc[0]["Model"]
            
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None
    
    def _load_metadata(self):
        """Load metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.models = json.load(f)
            else:
                self.models = {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            self.models = {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def export_model(self, model_name: str, export_path: str) -> bool:
        """Export a model to a specific path"""
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found in registry")
                return False
            
            model = self.load_model(model_name)
            if model is None:
                return False
            
            joblib.dump(model, export_path)
            logger.info(f"Model {model_name} exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_name}: {e}")
            return False
