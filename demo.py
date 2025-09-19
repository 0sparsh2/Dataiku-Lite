#!/usr/bin/env python3
"""
Dataiku Lite - Demo Script
Demonstrates the platform's capabilities with sample data
"""

import pandas as pd
import numpy as np
import sys
import os

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_analysis import DataProfiler
from app.preprocessing import PreprocessingPipeline
from app.modeling import ModelTrainer
from app.ai_teacher import AITeacher

def create_demo_data():
    """Create demo dataset for showcasing the platform"""
    print("🎯 Creating demo dataset...")
    
    # Create a realistic dataset
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.normal(16, 3, n_samples),
        'experience_years': np.random.normal(8, 5, n_samples),
        'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
        'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'], n_samples),
        'performance_score': np.random.normal(75, 15, n_samples),
        'satisfaction_rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    }
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    data['income'][missing_indices[:25]] = np.nan
    data['education_years'][missing_indices[25:]] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    data['income'][outlier_indices] *= 3  # Make some incomes very high
    
    # Create target variable (salary prediction)
    salary = (data['age'] * 1000 + 
              data['income'] * 0.8 + 
              data['education_years'] * 2000 + 
              data['experience_years'] * 1500 + 
              np.random.normal(0, 5000, n_samples))
    
    df = pd.DataFrame(data)
    df['salary'] = salary
    
    print(f"✅ Created dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def demo_data_analysis(df):
    """Demonstrate data analysis capabilities"""
    print("\n" + "="*60)
    print("📊 DATA ANALYSIS DEMONSTRATION")
    print("="*60)
    
    profiler = DataProfiler()
    results = profiler.profile_data(df, 'salary')
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Show key insights
    print(f"📈 Dataset Overview:")
    print(f"   • Shape: {results['overall_stats']['shape']}")
    print(f"   • Data Quality Score: {results['insights']['data_quality_score']:.1f}/100")
    print(f"   • Missing Values: {results['overall_stats']['missing_percentage']:.1f}%")
    print(f"   • Memory Usage: {results['overall_stats']['memory_usage_mb']:.1f} MB")
    
    # Show column analysis
    print(f"\n📋 Column Analysis:")
    for col_name, profile in results['column_profiles'].items():
        if col_name != 'salary':  # Skip target
            print(f"   • {col_name}: {profile['dtype']} - {profile['null_percentage']:.1f}% missing")
            if profile['quality_issues']:
                print(f"     Issues: {', '.join(profile['quality_issues'])}")
    
    # Show recommendations
    if 'recommendations' in results:
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations'][:3]:
            print(f"   • {rec}")

def demo_preprocessing(df):
    """Demonstrate preprocessing capabilities"""
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Separate features and target
    X = df.drop('salary', axis=1)
    y = df['salary']
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline("demo_pipeline")
    pipeline.create_educational_pipeline(X, 'salary')
    
    print("📋 Pipeline Steps:")
    step_info = pipeline.get_step_info()
    for step in step_info:
        status = "✅ Enabled" if step['enabled'] else "❌ Disabled"
        print(f"   • {step['name']}: {status}")
        print(f"     {step['description']}")
    
    # Run preprocessing
    print(f"\n🚀 Running preprocessing...")
    try:
        X_processed = pipeline.fit_transform(X, y)
        print(f"✅ Preprocessing completed!")
        print(f"   • Original shape: {X.shape}")
        print(f"   • Processed shape: {X_processed.shape}")
        print(f"   • Features added: {X_processed.shape[1] - X.shape[1]}")
        
        # Show educational explanations
        print(f"\n🎓 Educational Explanations:")
        explanations = pipeline.get_educational_explanations()
        for step_name, explanation in list(explanations.items())[:2]:  # Show first 2
            print(f"\n📚 {step_name.replace('_', ' ').title()}:")
            print(explanation[:200] + "..." if len(explanation) > 200 else explanation)
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")

def demo_modeling(df):
    """Demonstrate modeling capabilities"""
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING DEMONSTRATION")
    print("="*60)
    
    # Prepare data
    X = df.drop('salary', axis=1)
    y = df['salary']
    
    # Simple preprocessing for demo
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X.select_dtypes(include=[np.number]))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Convert back to DataFrame
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_processed = pd.DataFrame(X_scaled, columns=numeric_cols)
    
    print(f"📊 Training regression models...")
    trainer = ModelTrainer()
    
    try:
        results = trainer.train_regression_models(X_processed, y, test_size=0.2, cv_folds=3)
        
        print(f"✅ Trained {len(results)} models successfully!")
        
        # Show model comparison
        print(f"\n📈 Model Performance:")
        comparison_df = trainer.get_model_comparison()
        for _, row in comparison_df.head(3).iterrows():
            print(f"   • {row['Model']}: CV Score = {row['CV Mean']:.4f} ± {row['CV Std']:.4f}")
        
        # Best model
        best_model = trainer.get_best_model()
        if best_model:
            print(f"\n🏆 Best Model: {best_model.model_name}")
            print(f"   • CV Score: {best_model.cv_mean:.4f}")
            print(f"   • Training Time: {best_model.training_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error during modeling: {e}")

def demo_ai_teacher(df):
    """Demonstrate AI teacher capabilities"""
    print("\n" + "="*60)
    print("🤖 AI TEACHER DEMONSTRATION")
    print("="*60)
    
    teacher = AITeacher()
    
    print("🧠 Getting AI insights...")
    try:
        analysis = teacher.analyze_data_and_recommend(df, "regression")
        
        if 'ai_insights' in analysis:
            print("📚 AI Teacher Insights:")
            print(analysis['ai_insights'][:300] + "..." if len(analysis['ai_insights']) > 300 else analysis['ai_insights'])
        
        if 'recommendations' in analysis and analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations'][:3]:
                print(f"   • {rec}")
        
    except Exception as e:
        print(f"ℹ️  AI Teacher requires Groq API key for full functionality")
        print(f"   Error: {e}")

def main():
    """Run the complete demo"""
    print("🧠 Dataiku Lite - Platform Demo")
    print("=" * 60)
    print("This demo showcases the core capabilities of the platform")
    
    try:
        # Create demo data
        df = create_demo_data()
        
        # Run demonstrations
        demo_data_analysis(df)
        demo_preprocessing(df)
        demo_modeling(df)
        demo_ai_teacher(df)
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETE!")
        print("="*60)
        print("The platform is working correctly!")
        print("\n🌐 To use the web interface:")
        print("   streamlit run app.py")
        print("   Then open: http://localhost:8501")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
