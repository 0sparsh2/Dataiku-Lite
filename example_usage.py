"""
Example usage of Dataiku Lite platform
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import sys
import os

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_analysis import DataProfiler
from app.preprocessing import PreprocessingPipeline
from app.modeling import ModelTrainer
from app.ai_teacher import AITeacher

def create_sample_data():
    """Create sample datasets for demonstration"""
    print("Creating sample datasets...")
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Add some missing values and categorical features
    X_class_df = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(10)])
    X_class_df['categorical'] = np.random.choice(['A', 'B', 'C'], 1000)
    X_class_df.loc[100:150, 'feature_0'] = np.nan  # Add missing values
    y_class_series = pd.Series(y_class, name='target')
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=8,
        noise=0.1,
        random_state=42
    )
    X_reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(8)])
    y_reg_series = pd.Series(y_reg, name='target')
    
    return {
        'classification': (X_class_df, y_class_series),
        'regression': (X_reg_df, y_reg_series)
    }

def demonstrate_data_analysis():
    """Demonstrate data analysis capabilities"""
    print("\n" + "="*60)
    print("📊 DATA ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Initialize profiler
    profiler = DataProfiler()
    
    # Profile the data
    print("Profiling classification dataset...")
    results = profiler.profile_data(X, y.name)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    # Display results
    print(f"\nDataset Shape: {results['overall_stats']['shape']}")
    print(f"Data Quality Score: {results['insights']['data_quality_score']:.1f}/100")
    print(f"Missing Values: {results['overall_stats']['missing_percentage']:.1f}%")
    print(f"Memory Usage: {results['overall_stats']['memory_usage_mb']:.1f} MB")
    
    # Show column analysis
    print(f"\nColumn Analysis:")
    for col_name, profile in results['column_profiles'].items():
        print(f"  {col_name}: {profile['dtype']} - {profile['null_percentage']:.1f}% missing")
        if profile['quality_issues']:
            print(f"    Issues: {', '.join(profile['quality_issues'])}")

def demonstrate_preprocessing():
    """Demonstrate preprocessing capabilities"""
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Create preprocessing pipeline
    print("Creating educational preprocessing pipeline...")
    pipeline = PreprocessingPipeline("demo_pipeline")
    pipeline.create_educational_pipeline(X, y.name)
    
    # Show pipeline steps
    print(f"\nPipeline Steps:")
    step_info = pipeline.get_step_info()
    for step in step_info:
        status = "✅ Enabled" if step['enabled'] else "❌ Disabled"
        print(f"  {step['name']}: {status}")
        print(f"    {step['description']}")
    
    # Run preprocessing
    print(f"\nRunning preprocessing...")
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"Original shape: {X.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Features added: {X_processed.shape[1] - X.shape[1]}")

def demonstrate_modeling():
    """Demonstrate modeling capabilities"""
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Preprocess data first
    pipeline = PreprocessingPipeline("demo_pipeline")
    pipeline.create_educational_pipeline(X, y.name)
    X_processed = pipeline.fit_transform(X, y)
    
    # Train models
    print("Training classification models...")
    trainer = ModelTrainer()
    results = trainer.train_classification_models(X_processed, y, test_size=0.2, cv_folds=3)
    
    # Show results
    print(f"\nTrained {len(results)} models:")
    comparison_df = trainer.get_model_comparison()
    print(comparison_df[['Model', 'CV Mean', 'CV Std', 'Training Time']].to_string(index=False))
    
    # Best model
    best_model = trainer.get_best_model()
    if best_model:
        print(f"\nBest Model: {best_model.model_name}")
        print(f"CV Score: {best_model.cv_mean:.4f} ± {best_model.cv_std:.4f}")

def demonstrate_ai_teacher():
    """Demonstrate AI teacher capabilities"""
    print("\n" + "="*60)
    print("🤖 AI TEACHER DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Initialize AI teacher
    teacher = AITeacher()
    
    # Analyze data and get recommendations
    print("Getting AI insights for data analysis...")
    try:
        analysis = teacher.analyze_data_and_recommend(X, "classification")
        
        if 'ai_insights' in analysis:
            print("AI Teacher Insights:")
            print(analysis['ai_insights'][:500] + "..." if len(analysis['ai_insights']) > 500 else analysis['ai_insights'])
        
        if 'recommendations' in analysis and analysis['recommendations']:
            print("\nRecommendations:")
            for rec in analysis['recommendations'][:3]:  # Show first 3
                print(f"  • {rec}")
    
    except Exception as e:
        print(f"Note: AI Teacher requires Groq API key. Error: {e}")
        print("To use AI features, add your Groq API key to config.py")

def main():
    """Run all demonstrations"""
    print("🧠 Dataiku Lite - Example Usage")
    print("This script demonstrates the core capabilities of the platform")
    
    try:
        demonstrate_data_analysis()
        demonstrate_preprocessing()
        demonstrate_modeling()
        demonstrate_ai_teacher()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*60)
        print("The platform is working correctly!")
        print("\nTo use the web interface, run:")
        print("streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
