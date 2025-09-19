"""
Dataiku Lite - ML/DS Educational Platform
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import sys
import os
import io

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.data_analysis import DataProfiler
from app.preprocessing import PreprocessingPipeline
from app.modeling import ModelTrainer
from app.ai_teacher import AITeacher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_arff_file(uploaded_file):
    """Load ARFF file and convert to pandas DataFrame"""
    try:
        # Try using scipy.io.arff for better ARFF support
        try:
            from scipy.io import arff
            import tempfile
            
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.arff', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Load ARFF file using scipy
            data, meta = arff.loadarff(tmp_file_path)
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            # Convert bytes to strings for categorical data
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace("b'", "").str.replace("'", "")
            
            return df
            
        except ImportError:
            # Fallback to simple parser if scipy is not available
            return load_arff_simple(uploaded_file)
            
    except Exception as e:
        st.error(f"Error loading ARFF file: {str(e)}")
        return pd.DataFrame()

def load_arff_simple(uploaded_file):
    """Simple ARFF parser fallback"""
    try:
        # Read the ARFF file content
        content = uploaded_file.read().decode('utf-8')
        
        # Simple ARFF parser
        lines = content.split('\n')
        data_lines = []
        attributes = {}
        in_data_section = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('@ATTRIBUTE'):
                # Parse attribute definition
                parts = line.split()
                if len(parts) >= 3:
                    attr_name = parts[1]
                    attr_type = parts[2].upper()
                    attributes[attr_name] = attr_type
            elif line.upper().startswith('@DATA'):
                in_data_section = True
                continue
            elif in_data_section and line and not line.startswith('%'):
                # Parse data line
                data_lines.append(line.split(','))
        
        # Create DataFrame
        if data_lines and attributes:
            # Get attribute names in order
            attr_names = list(attributes.keys())
            
            # Create DataFrame
            df = pd.DataFrame(data_lines, columns=attr_names)
            
            # Convert data types
            for col, attr_type in attributes.items():
                if col in df.columns:
                    if attr_type == 'NUMERIC' or attr_type == 'REAL':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif attr_type == 'INTEGER':
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    # For nominal/string types, keep as string
            
            return df
        else:
            st.error("Could not parse ARFF file. Please check the file format.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in simple ARFF parser: {str(e)}")
        return pd.DataFrame()

# Page configuration
st.set_page_config(
    page_title="Dataiku Lite - ML/DS Educational Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Dataiku Lite</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6c757d;">Educational Machine Learning & Data Science Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Data Analysis", "🔧 Preprocessing", "🤖 Model Training", "📈 Results", "🎓 Learn"]
        )
        
        st.header("⚙️ Settings")
        st.text_input("Groq API Key", key="groq_api_key", type="password", help="Enter your Groq API key for AI explanations")
        
        if st.button("🔧 Configure Settings"):
            st.success("Settings saved!")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'profiler' not in st.session_state:
        st.session_state.profiler = DataProfiler()
    if 'ai_teacher' not in st.session_state:
        st.session_state.ai_teacher = AITeacher()
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = PreprocessingPipeline()
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ModelTrainer()
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "🔧 Preprocessing":
        preprocessing_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "📈 Results":
        results_page()
    elif page == "🎓 Learn":
        learn_page()

def home_page():
    """Home page with overview and data upload"""
    st.markdown('<div class="section-header">Welcome to Dataiku Lite!</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Dataiku Lite** is your comprehensive educational platform for machine learning and data science. 
        Whether you're a beginner or an experienced practitioner, our AI-powered platform will guide you 
        through every step of your data science journey.
        
        ### 🚀 Key Features:
        - **Smart Data Analysis**: Automatic profiling with AI insights
        - **Educational Preprocessing**: Step-by-step guidance with explanations
        - **Intelligent Model Training**: AI-recommended algorithms and parameters
        - **Interactive Visualizations**: Dynamic charts and dashboards
        - **AI Teacher**: Get explanations and recommendations for every step
        
        ### 📚 Perfect for:
        - Learning data science concepts
        - Understanding ML workflows
        - Experimenting with different approaches
        - Building production-ready models
        """)
    
    with col2:
        st.markdown("### 🎯 Quick Start")
        st.markdown("""
        1. **Upload your data** (CSV, Excel, JSON)
        2. **Explore and analyze** with AI guidance
        3. **Preprocess** with educational explanations
        4. **Train models** with smart recommendations
        5. **Evaluate and iterate** for better results
        """)
    
    # Data upload section
    st.markdown('<div class="section-header">📁 Upload Your Data</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'arff'],
        help="Upload your dataset to get started"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type with proper encoding handling
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for CSV files
                try:
                    data = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        data = pd.read_csv(uploaded_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        data = pd.read_csv(uploaded_file, encoding='cp1252')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.arff'):
                data = load_arff_file(uploaded_file)
            
            st.session_state.data = data
            
            st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{data.shape[0]:,}")
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show sample data
            st.markdown("### 📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
        except UnicodeDecodeError as e:
            st.error(f"❌ Encoding error: {str(e)}")
            st.info("💡 Try saving your file with UTF-8 encoding or use a different file format.")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("💡 Supported formats: CSV, Excel (.xlsx/.xls), JSON, ARFF")

def data_analysis_page():
    """Data analysis and profiling page"""
    st.markdown('<div class="section-header">📊 Data Analysis & Profiling</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the Home page")
        return
    
    data = st.session_state.data
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    with col1:
        run_analysis = st.button("🔍 Run Full Analysis", type="primary")
    with col2:
        show_ai_insights = st.button("🤖 Get AI Insights")
    with col3:
        export_report = st.button("📄 Export Report")
    
    if run_analysis:
        with st.spinner("Analyzing data..."):
            try:
                # Run profiling
                profile_results = st.session_state.profiler.profile_data(data)
                
                if "error" in profile_results:
                    st.error(f"Analysis error: {profile_results['error']}")
                    return
                
                st.session_state.profile_results = profile_results
                st.success("✅ Analysis completed!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    # Display results if available
    if 'profile_results' in st.session_state:
        results = st.session_state.profile_results
        
        # Overall statistics
        st.markdown("### 📈 Overall Statistics")
        overall_stats = results['overall_stats']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Quality Score", f"{results['insights']['data_quality_score']:.1f}/100")
        with col2:
            st.metric("Missing Values", f"{overall_stats['missing_percentage']:.1f}%")
        with col3:
            st.metric("Duplicate Rows", f"{overall_stats['duplicate_percentage']:.1f}%")
        with col4:
            st.metric("Memory Usage", f"{overall_stats['memory_usage_mb']:.1f} MB")
        
        # Data quality insights
        if results['insights']['key_findings']:
            st.markdown("### 🔍 Key Findings")
            for finding in results['insights']['key_findings']:
                st.markdown(f"• {finding}")
        
        # Column analysis
        st.markdown("### 📋 Column Analysis")
        column_profiles = results['column_profiles']
        
        # Create a summary table
        summary_data = []
        for col_name, profile in column_profiles.items():
            summary_data.append({
                "Column": col_name,
                "Type": profile['dtype'],
                "Missing %": f"{profile['null_percentage']:.1f}%",
                "Unique %": f"{profile['unique_percentage']:.1f}%",
                "Issues": len(profile['quality_issues']),
                "Is Numeric": profile['is_numeric'],
                "Is Categorical": profile['is_categorical']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Detailed column analysis
        selected_column = st.selectbox("Select column for detailed analysis:", data.columns)
        if selected_column:
            profile = column_profiles[selected_column]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{selected_column} Details:**")
                st.json({
                    "Data Type": profile['dtype'],
                    "Missing Values": f"{profile['null_count']} ({profile['null_percentage']:.1f}%)",
                    "Unique Values": f"{profile['unique_count']} ({profile['unique_percentage']:.1f}%)",
                    "Is Numeric": profile['is_numeric'],
                    "Is Categorical": profile['is_categorical']
                })
                
                if profile['quality_issues']:
                    st.markdown("**Quality Issues:**")
                    for issue in profile['quality_issues']:
                        st.markdown(f"⚠️ {issue}")
            
            with col2:
                # Show statistics if available
                if profile['statistics']:
                    st.markdown("**Statistics:**")
                    stats_df = pd.DataFrame(list(profile['statistics'].items()), 
                                          columns=['Metric', 'Value'])
                    st.dataframe(stats_df, use_container_width=True)
        
        # Visualizations
        st.markdown("### 📊 Visualizations")
        
        # Missing values heatmap
        if overall_stats['missing_cells'] > 0:
            st.markdown("**Missing Values Heatmap:**")
            missing_data = data.isnull()
            fig = px.imshow(missing_data.T, 
                          labels=dict(x="Row", y="Column", color="Missing"),
                          title="Missing Values Pattern")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("**Correlation Heatmap:**")
            corr_matrix = data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, 
                          text_auto=True,
                          title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    if show_ai_insights and st.session_state.data is not None:
        with st.spinner("Getting AI insights..."):
            try:
                # Get AI insights
                ai_analysis = st.session_state.ai_teacher.analyze_data_and_recommend(
                    st.session_state.data
                )
                
                st.markdown("### 🤖 AI Teacher Insights")
                st.markdown(ai_analysis['ai_insights'])
                
                if ai_analysis['recommendations']:
                    st.markdown("### 💡 Recommendations")
                    for rec in ai_analysis['recommendations']:
                        st.markdown(f"• {rec}")
                
            except Exception as e:
                st.error(f"❌ Error getting AI insights: {str(e)}")

def preprocessing_page():
    """Data preprocessing page"""
    st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first from the Home page")
        return
    
    data = st.session_state.data
    
    # Preprocessing options
    st.markdown("### ⚙️ Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Preprocessing Steps:**")
        handle_missing = st.checkbox("Handle Missing Values", value=True)
        handle_outliers = st.checkbox("Handle Outliers", value=True)
        encode_categorical = st.checkbox("Encode Categorical Variables", value=True)
        scale_features = st.checkbox("Scale Features", value=True)
        engineer_features = st.checkbox("Feature Engineering", value=True)
    
    with col2:
        st.markdown("**Target Column (for supervised learning):**")
        target_column = st.selectbox("Select target column:", ["None"] + list(data.columns))
        if target_column == "None":
            target_column = None
    
    # Create and configure pipeline
    if st.button("🔧 Create Preprocessing Pipeline", type="primary"):
        with st.spinner("Creating preprocessing pipeline..."):
            try:
                # Create educational pipeline
                pipeline = PreprocessingPipeline("educational_pipeline")
                pipeline.create_educational_pipeline(data, target_column)
                
                # Configure based on user selections
                if not handle_missing:
                    pipeline.disable_step("missing_values")
                if not handle_outliers:
                    pipeline.disable_step("outliers")
                if not encode_categorical:
                    pipeline.disable_step("categorical_encoding")
                if not scale_features:
                    pipeline.disable_step("scaling")
                if not engineer_features:
                    pipeline.disable_step("feature_engineering")
                
                st.session_state.pipeline = pipeline
                st.success("✅ Preprocessing pipeline created!")
                
                # Show pipeline info
                st.markdown("### 📋 Pipeline Steps")
                step_info = pipeline.get_step_info()
                for step in step_info:
                    status = "✅ Enabled" if step['enabled'] else "❌ Disabled"
                    st.markdown(f"**{step['name']}** - {status}")
                    st.markdown(f"*{step['description']}*")
                
            except Exception as e:
                st.error(f"❌ Error creating pipeline: {str(e)}")
    
    # Run preprocessing
    if 'pipeline' in st.session_state and st.session_state.pipeline.steps:
        if st.button("🚀 Run Preprocessing", type="primary"):
            with st.spinner("Running preprocessing..."):
                try:
                    # Prepare target variable
                    y = None
                    if target_column and target_column in data.columns:
                        y = data[target_column]
                        X = data.drop(columns=[target_column])
                    else:
                        X = data
                    
                    # Fit and transform
                    X_processed = st.session_state.pipeline.fit_transform(X, y)
                    
                    st.session_state.processed_data = X_processed
                    st.session_state.target_data = y
                    
                    st.success("✅ Preprocessing completed!")
                    
                    # Show results
                    st.markdown("### 📊 Preprocessing Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Shape", f"{X.shape[0]} × {X.shape[1]}")
                    with col2:
                        st.metric("Processed Shape", f"{X_processed.shape[0]} × {X_processed.shape[1]}")
                    
                    # Show sample of processed data
                    st.markdown("**Processed Data Preview:**")
                    st.dataframe(X_processed.head(10), use_container_width=True)
                    
                    # Show educational explanations
                    st.markdown("### 🎓 Educational Explanations")
                    explanations = st.session_state.pipeline.get_educational_explanations()
                    
                    for step_name, explanation in explanations.items():
                        with st.expander(f"📚 {step_name.replace('_', ' ').title()}"):
                            st.markdown(explanation)
                
                except Exception as e:
                    st.error(f"❌ Error during preprocessing: {str(e)}")

def model_training_page():
    """Model training page"""
    st.markdown('<div class="section-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if 'processed_data' not in st.session_state:
        st.warning("⚠️ Please run preprocessing first")
        return
    
    X = st.session_state.processed_data
    y = st.session_state.target_data
    
    # Problem type detection and selection
    st.markdown("### 🎯 Problem Type")
    
    # Auto-detect problem type
    if y is not None:
        trainer = ModelTrainer()
        problem_analysis = trainer.detect_problem_type(y)
        
        # Show analysis
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Values", problem_analysis["unique_values"])
            st.metric("Min Samples/Class", problem_analysis["min_samples_per_class"])
        with col2:
            st.metric("Is Numeric", "Yes" if problem_analysis["is_numeric"] else "No")
            st.metric("Suggested Type", problem_analysis["suggested_type"])
        
        # Show warnings
        if problem_analysis["warnings"]:
            for warning in problem_analysis["warnings"]:
                st.warning(f"⚠️ {warning}")
        
        # Problem type selection with suggestion
        suggested_type = problem_analysis["suggested_type"]
        problem_type = st.selectbox(
            "What type of problem are you solving?",
            ["Classification", "Regression", "Clustering"],
            index=["Classification", "Regression", "Clustering"].index(suggested_type) if suggested_type in ["Classification", "Regression", "Clustering"] else 0
        )
        
        # Show additional info
        if problem_type == "Classification" and problem_analysis["min_samples_per_class"] < 2:
            st.error("❌ Classification requires at least 2 samples per class. Consider using Regression or Clustering instead.")
    else:
        problem_type = st.selectbox(
            "What type of problem are you solving?",
            ["Classification", "Regression", "Clustering"]
        )
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    with col2:
        random_state = st.number_input("Random State", value=42, min_value=0)
        max_models = st.slider("Max Models to Train", 3, 10, 5)
    
    # Train models
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            try:
                trainer = ModelTrainer(random_state=random_state)
                
                if problem_type == "Classification" and y is not None:
                    # Check if classification is possible
                    min_samples = y.value_counts().min()
                    if min_samples < 2:
                        st.error(f"❌ Classification not possible: minimum 2 samples per class required, but found {min_samples} samples in smallest class.")
                        st.info("💡 Try using Regression or Clustering instead.")
                        return
                    results = trainer.train_classification_models(X, y, test_size, cv_folds)
                elif problem_type == "Regression" and y is not None:
                    results = trainer.train_regression_models(X, y, test_size, cv_folds)
                elif problem_type == "Clustering":
                    results = trainer.train_clustering_models(X)
                else:
                    st.error("❌ Target variable required for supervised learning")
                    return
                
                st.session_state.training_results = results
                st.session_state.trainer = trainer
                
                st.success(f"✅ Trained {len(results)} models successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
    
    # Display results
    if 'training_results' in st.session_state:
        results = st.session_state.training_results
        
        # Model comparison
        st.markdown("### 📊 Model Comparison")
        comparison_df = st.session_state.trainer.get_model_comparison()
        st.dataframe(comparison_df, use_container_width=True)
        
        # Best model info
        best_model = st.session_state.trainer.get_best_model()
        if best_model:
            st.markdown("### 🏆 Best Model")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", best_model.model_name)
            with col2:
                st.metric("CV Score", f"{best_model.cv_mean:.4f}")
            with col3:
                st.metric("Training Time", f"{best_model.training_time:.2f}s")
            
            # Feature importance if available
            if best_model.feature_importance:
                st.markdown("### 📈 Feature Importance")
                importance_df = pd.DataFrame(
                    list(best_model.feature_importance.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df.head(10), 
                           x='Importance', y='Feature',
                           title="Top 10 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)
        
        # AI evaluation
        if st.button("🤖 Get AI Model Evaluation"):
            with st.spinner("Getting AI evaluation..."):
                try:
                    if best_model:
                        model_results = {
                            "model_type": best_model.model_name,
                            "metrics": {
                                "train_score": best_model.train_score,
                                "test_score": best_model.test_score,
                                "cv_mean": best_model.cv_mean,
                                "cv_std": best_model.cv_std
                            },
                            "training_time": best_model.training_time
                        }
                        
                        ai_evaluation = st.session_state.ai_teacher.evaluate_model_results(model_results)
                        st.markdown("### 🤖 AI Teacher Evaluation")
                        st.markdown(ai_evaluation)
                
                except Exception as e:
                    st.error(f"❌ Error getting AI evaluation: {str(e)}")

def results_page():
    """Results and visualization page"""
    st.markdown('<div class="section-header">📈 Results & Visualizations</div>', unsafe_allow_html=True)
    
    if 'training_results' not in st.session_state:
        st.warning("⚠️ Please train models first")
        return
    
    # Results summary
    st.markdown("### 📊 Results Summary")
    
    # Add more detailed visualizations and analysis here
    st.info("📈 Detailed results and visualizations will be displayed here")

def learn_page():
    """Learning and educational content page"""
    st.markdown('<div class="section-header">🎓 Learn Data Science</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📚 Educational Content
    
    Welcome to the learning section! Here you'll find comprehensive guides and explanations
    for various data science concepts and techniques.
    
    #### 🔍 Data Analysis
    - Understanding data types and distributions
    - Identifying data quality issues
    - Statistical analysis and visualization
    
    #### 🔧 Preprocessing
    - Handling missing values
    - Outlier detection and treatment
    - Feature encoding and scaling
    - Feature engineering techniques
    
    #### 🤖 Machine Learning
    - Model selection and evaluation
    - Hyperparameter tuning
    - Cross-validation strategies
    - Model interpretability
    
    #### 📊 Visualization
    - Creating effective visualizations
    - Choosing the right chart types
    - Interactive dashboards
    
    ### 💡 Tips & Best Practices
    - Always start with data exploration
    - Validate your preprocessing steps
    - Use cross-validation for model evaluation
    - Document your process and decisions
    - Iterate and improve continuously
    """)
    
    # Interactive Q&A section
    st.markdown("### ❓ Ask the AI Teacher")
    
    question = st.text_area("Ask any question about data science or your current project:")
    
    if st.button("🤖 Get Answer"):
        if question:
            with st.spinner("AI Teacher is thinking..."):
                try:
                    # This would integrate with the AI teacher for Q&A
                    st.info("AI Teacher Q&A feature coming soon!")
                except Exception as e:
                    st.error(f"❌ Error getting answer: {str(e)}")
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()
