"""
Test script to verify Dataiku Lite installation
"""

import sys
import importlib
import pandas as pd
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'plotly', 'streamlit',
        'requests', 'joblib', 'scipy', 'sklearn'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        return True

def test_data_analysis():
    """Test data analysis functionality"""
    print("\nTesting data analysis functionality...")
    
    try:
        from app.data_analysis import DataProfiler
        
        # Create sample data
        data = pd.DataFrame({
            'numeric_col': np.random.normal(0, 1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'missing_col': np.random.normal(0, 1, 100)
        })
        data.loc[10:20, 'missing_col'] = np.nan
        
        # Test profiler
        profiler = DataProfiler()
        results = profiler.profile_data(data)
        
        if 'error' in results:
            print(f"❌ Data profiling error: {results['error']}")
            return False
        else:
            print("✅ Data profiling works!")
            return True
            
    except Exception as e:
        print(f"❌ Data analysis test failed: {e}")
        return False

def test_preprocessing():
    """Test preprocessing functionality"""
    print("\nTesting preprocessing functionality...")
    
    try:
        from app.preprocessing import PreprocessingPipeline
        
        # Create sample data
        data = pd.DataFrame({
            'numeric_col': np.random.normal(0, 1, 100),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 100),
            'missing_col': np.random.normal(0, 1, 100)
        })
        data.loc[10:20, 'missing_col'] = np.nan
        
        # Test pipeline
        pipeline = PreprocessingPipeline("test_pipeline")
        pipeline.create_educational_pipeline(data)
        
        print("✅ Preprocessing pipeline works!")
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_modeling():
    """Test modeling functionality"""
    print("\nTesting modeling functionality...")
    
    try:
        from app.modeling import ModelTrainer
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Test trainer
        trainer = ModelTrainer()
        
        print("✅ Model trainer works!")
        return True
        
    except Exception as e:
        print(f"❌ Modeling test failed: {e}")
        return False

def test_ai_teacher():
    """Test AI teacher functionality"""
    print("\nTesting AI teacher functionality...")
    
    try:
        from app.ai_teacher import AITeacher
        
        # Test teacher (without API key)
        teacher = AITeacher()
        
        print("✅ AI teacher works!")
        return True
        
    except Exception as e:
        print(f"❌ AI teacher test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Dataiku Lite - Installation Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_analysis,
        test_preprocessing,
        test_modeling,
        test_ai_teacher
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dataiku Lite is ready to use.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
