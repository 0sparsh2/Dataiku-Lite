"""
AI Teacher - Main class for providing educational guidance and explanations
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import requests
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class TeachingContext:
    """Context for AI teaching interactions"""
    user_skill_level: str = "beginner"  # beginner, intermediate, advanced
    problem_type: Optional[str] = None  # classification, regression, clustering, etc.
    data_characteristics: Dict[str, Any] = None
    current_step: str = ""
    previous_steps: List[str] = None

class AITeacher:
    """
    AI Teacher that provides intelligent explanations and recommendations
    using Groq API with Qwen3-32B model
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = settings.groq_model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.context = TeachingContext()
        
    def _call_groq_api(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """Call Groq API with error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": 2000,
                "top_p": 1,
                "reasoning_effort": "medium",
                "stream": False
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            return "I'm having trouble connecting to the AI service. Please check your API key and try again."
        except Exception as e:
            logger.error(f"Unexpected error in Groq API call: {e}")
            return "An unexpected error occurred. Please try again."
    
    def analyze_data_and_recommend(self, data: pd.DataFrame, problem_type: str = None) -> Dict[str, Any]:
        """
        Analyze data characteristics and provide educational recommendations
        """
        try:
            # Basic data analysis
            data_info = self._analyze_data_characteristics(data)
            
            # Generate AI-powered insights
            analysis_prompt = self._create_data_analysis_prompt(data_info, problem_type)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert data science educator. Provide clear, educational explanations of data analysis results and actionable recommendations. Always explain WHY each recommendation is important."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ]
            
            ai_insights = self._call_groq_api(messages)
            
            return {
                "data_characteristics": data_info,
                "ai_insights": ai_insights,
                "recommendations": self._extract_recommendations(ai_insights),
                "next_steps": self._suggest_next_steps(data_info, problem_type)
            }
            
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return {"error": str(e)}
    
    def explain_preprocessing_step(self, step_name: str, data_context: Dict[str, Any]) -> str:
        """
        Provide detailed explanation of a preprocessing step
        """
        try:
            explanation_prompt = f"""
            Explain the preprocessing step '{step_name}' in the context of this data:
            
            Data shape: {data_context.get('shape', 'Unknown')}
            Missing values: {data_context.get('missing_values', 'Unknown')}
            Data types: {data_context.get('dtypes', 'Unknown')}
            Problem type: {data_context.get('problem_type', 'Unknown')}
            
            Provide:
            1. What this step does
            2. Why it's important for this specific data
            3. How it affects the data
            4. Potential alternatives
            5. Common mistakes to avoid
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a data science educator. Provide clear, step-by-step explanations that help users understand not just what to do, but why it matters."
                },
                {
                    "role": "user",
                    "content": explanation_prompt
                }
            ]
            
            return self._call_groq_api(messages)
            
        except Exception as e:
            logger.error(f"Error explaining preprocessing step: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def evaluate_model_results(self, model_results: Dict[str, Any]) -> str:
        """
        Interpret model results and provide educational feedback
        """
        try:
            evaluation_prompt = f"""
            Analyze these model results and provide educational feedback:
            
            Model type: {model_results.get('model_type', 'Unknown')}
            Metrics: {model_results.get('metrics', {})}
            Training time: {model_results.get('training_time', 'Unknown')}
            Cross-validation scores: {model_results.get('cv_scores', [])}
            
            Provide:
            1. Interpretation of the metrics in plain language
            2. Whether the model is overfitting, underfitting, or well-fitted
            3. Specific areas for improvement
            4. What these results mean for the business problem
            5. Next steps to improve performance
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a machine learning expert and educator. Help users understand their model performance and how to improve it."
                },
                {
                    "role": "user",
                    "content": evaluation_prompt
                }
            ]
            
            return self._call_groq_api(messages)
            
        except Exception as e:
            logger.error(f"Error evaluating model results: {e}")
            return f"Error generating evaluation: {str(e)}"
    
    def recommend_model(self, problem_type: str, data_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend appropriate models based on problem type and data characteristics
        """
        try:
            recommendation_prompt = f"""
            Recommend machine learning models for this problem:
            
            Problem type: {problem_type}
            Data size: {data_characteristics.get('shape', 'Unknown')}
            Features: {data_characteristics.get('n_features', 'Unknown')}
            Target distribution: {data_characteristics.get('target_distribution', 'Unknown')}
            Missing values: {data_characteristics.get('missing_percentage', 'Unknown')}%
            
            Provide 3-5 model recommendations with:
            1. Model name and type
            2. Why it's suitable for this problem
            3. Pros and cons
            4. Expected performance characteristics
            5. Implementation complexity
            """
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a machine learning expert. Provide practical model recommendations with clear explanations of when and why to use each model."
                },
                {
                    "role": "user",
                    "content": recommendation_prompt
                }
            ]
            
            ai_response = self._call_groq_api(messages)
            return self._parse_model_recommendations(ai_response)
            
        except Exception as e:
            logger.error(f"Error recommending models: {e}")
            return []
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic data characteristics"""
        try:
            info = {
                "shape": data.shape,
                "n_features": len(data.columns),
                "dtypes": data.dtypes.value_counts().to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "missing_percentage": (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
                "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist(),
                "memory_usage": data.memory_usage(deep=True).sum() / 1024**2,  # MB
            }
            
            # Add basic statistics for numeric columns
            if info["numeric_columns"]:
                info["numeric_stats"] = data[info["numeric_columns"]].describe().to_dict()
            
            return info
            
        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            return {"error": str(e)}
    
    def _create_data_analysis_prompt(self, data_info: Dict[str, Any], problem_type: str) -> str:
        """Create prompt for data analysis"""
        return f"""
        Analyze this dataset and provide educational insights:
        
        Dataset Overview:
        - Shape: {data_info.get('shape', 'Unknown')}
        - Features: {data_info.get('n_features', 'Unknown')}
        - Memory usage: {data_info.get('memory_usage', 0):.2f} MB
        - Missing values: {data_info.get('missing_percentage', 0):.2f}%
        
        Data Types:
        - Numeric: {len(data_info.get('numeric_columns', []))} columns
        - Categorical: {len(data_info.get('categorical_columns', []))} columns
        - DateTime: {len(data_info.get('datetime_columns', []))} columns
        
        Problem Type: {problem_type or 'Not specified'}
        
        Please provide:
        1. Key insights about data quality
        2. Potential issues to address
        3. Recommended preprocessing steps
        4. Suitable modeling approaches
        5. Educational tips for this type of data
        """
    
    def _extract_recommendations(self, ai_insights: str) -> List[str]:
        """Extract actionable recommendations from AI response"""
        # Simple extraction - in production, this could be more sophisticated
        lines = ai_insights.split('\n')
        recommendations = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                recommendations.append(line.strip())
        return recommendations[:5]  # Limit to top 5
    
    def _suggest_next_steps(self, data_info: Dict[str, Any], problem_type: str) -> List[str]:
        """Suggest next steps based on data analysis"""
        steps = []
        
        if data_info.get('missing_percentage', 0) > 5:
            steps.append("Handle missing values")
        
        if len(data_info.get('categorical_columns', [])) > 0:
            steps.append("Encode categorical variables")
        
        if len(data_info.get('numeric_columns', [])) > 10:
            steps.append("Consider feature selection")
        
        if problem_type in ['classification', 'regression']:
            steps.append("Split data into train/test sets")
            steps.append("Choose and train initial models")
        
        return steps
    
    def _parse_model_recommendations(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured model recommendations"""
        # This is a simplified parser - in production, use more sophisticated parsing
        recommendations = []
        
        # Simple keyword-based parsing
        models = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'Neural Network']
        
        for model in models:
            if model.lower() in ai_response.lower():
                recommendations.append({
                    "name": model,
                    "suitability": "High" if model in ['Random Forest', 'XGBoost'] else "Medium",
                    "complexity": "Low" if model in ['Logistic Regression'] else "Medium",
                    "explanation": f"{model} is suitable for this problem type"
                })
        
        return recommendations[:5]
    
    def update_context(self, **kwargs):
        """Update teaching context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def get_context(self) -> TeachingContext:
        """Get current teaching context"""
        return self.context
