# Dataiku Lite - Quick Start Guide

## 🚀 Getting Started

### 1. Installation
```bash
# Clone or download the project
cd dataiku_lite

# Run the setup script
python3 setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Edit config.py to add your Groq API key (optional)
# This enables AI teacher features
```

### 3. Run the Application
```bash
# Start the web interface
python3 run.py

# Or run directly with Streamlit
streamlit run app.py
```

### 4. Test Installation
```bash
# Run the test script
python3 test_installation.py

# Try the example usage
python3 example_usage.py
```

## 📊 Using the Platform

### Step 1: Upload Data
- Go to the Home page
- Upload CSV, Excel, or JSON files
- View basic data information

### Step 2: Data Analysis
- Navigate to "Data Analysis" page
- Click "Run Full Analysis" for comprehensive profiling
- Get AI insights (requires Groq API key)

### Step 3: Preprocessing
- Go to "Preprocessing" page
- Configure preprocessing steps
- Run the pipeline with educational explanations

### Step 4: Model Training
- Visit "Model Training" page
- Select problem type (Classification/Regression/Clustering)
- Train multiple models and compare performance

### Step 5: Results & Visualization
- Check "Results" page for detailed analysis
- View interactive charts and metrics
- Get AI evaluation of your models

## 🎓 Educational Features

### AI Teacher
- Explains every step with educational context
- Provides recommendations based on your data
- Answers questions about data science concepts

### Step-by-Step Guidance
- Each preprocessing step includes explanations
- Learn why each technique is important
- Understand best practices and common pitfalls

### Interactive Learning
- Visual feedback for every action
- Real-time explanations and warnings
- Progressive disclosure of complex concepts

## 🔧 Key Features

### Data Analysis
- Automatic data profiling
- Missing value analysis
- Distribution analysis
- Correlation analysis
- Data quality assessment

### Preprocessing
- Missing value handling
- Outlier detection and treatment
- Categorical encoding
- Feature scaling
- Feature engineering

### Model Training
- Multiple algorithm support
- Cross-validation
- Model comparison
- Feature importance analysis
- Learning curve visualization

### Visualization
- Interactive charts with Plotly
- Missing values heatmap
- Correlation matrix
- Distribution plots
- Model performance dashboards

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing Dependencies**
   ```bash
   pip install scikit-learn pandas numpy plotly streamlit
   ```

3. **Pydantic Settings Error**
   ```bash
   pip install pydantic-settings
   ```

4. **Streamlit Not Starting**
   ```bash
   streamlit run app.py --server.port 8501
   ```

### Getting Help

- Check the README.md for detailed documentation
- Run `python3 test_installation.py` to diagnose issues
- Look at `example_usage.py` for usage examples

## 📚 Next Steps

1. **Explore the Interface**: Familiarize yourself with all pages
2. **Try Different Datasets**: Upload various types of data
3. **Experiment with Models**: Try different algorithms and parameters
4. **Use AI Features**: Add your Groq API key for enhanced guidance
5. **Learn from Explanations**: Read the educational content for each step

## 🎯 Tips for Success

- Start with small, clean datasets
- Read the AI explanations carefully
- Experiment with different preprocessing steps
- Compare multiple models before choosing the best one
- Use the visualization tools to understand your data better

---

**Happy Learning! 🎉**

*Dataiku Lite - Making Data Science Accessible and Educational*
