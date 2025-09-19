# Dataiku Lite - ML/DS Educational Platform

A comprehensive educational machine learning and data science platform that combines powerful analytics with intelligent AI guidance. Built with Python, Streamlit, and integrated with Groq API for educational explanations.

## 🚀 Features

### Core Capabilities
- **Smart Data Analysis**: Automatic profiling with AI-powered insights
- **Educational Preprocessing**: Step-by-step guidance with detailed explanations
- **Intelligent Model Training**: AI-recommended algorithms and parameters
- **Interactive Visualizations**: Dynamic charts and dashboards
- **AI Teacher Integration**: Get explanations and recommendations for every step

### Educational Focus
- **Why, not just What**: Every action includes educational context
- **Progressive Learning**: Adapts to user skill level
- **Mistake Prevention**: Proactive warnings and guidance
- **Best Practices**: Built-in recommendations for different scenarios

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **Web Interface**: Streamlit
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **Data Processing**: Pandas, NumPy, Polars
- **Visualization**: Plotly, Seaborn, Matplotlib
- **AI Integration**: Groq API with Qwen3-32B
- **Hyperparameter Optimization**: Optuna

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dataiku_lite
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp config.py.example config.py
   # Edit config.py with your Groq API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🎯 Quick Start

1. **Upload Data**: Support for CSV, Excel, JSON files
2. **Explore**: Automatic data profiling with AI insights
3. **Preprocess**: Guided preprocessing with educational explanations
4. **Train**: AI-recommended model selection and training
5. **Evaluate**: Comprehensive model evaluation and interpretation

## 📚 Educational Modules

### Data Analysis
- Automatic data type detection
- Missing value analysis
- Distribution analysis
- Correlation analysis
- Data quality assessment

### Preprocessing
- Missing value handling strategies
- Outlier detection and treatment
- Categorical encoding methods
- Feature scaling and normalization
- Feature engineering techniques

### Model Training
- Classification algorithms
- Regression algorithms
- Clustering algorithms
- Cross-validation strategies
- Hyperparameter optimization

### Evaluation
- Comprehensive metrics
- Model comparison
- Feature importance analysis
- Learning curve analysis
- Model interpretability

## 🤖 AI Teacher Features

The AI Teacher provides intelligent guidance throughout the entire workflow:

- **Data Analysis Insights**: Explains data characteristics and quality issues
- **Preprocessing Guidance**: Recommends appropriate techniques with explanations
- **Model Recommendations**: Suggests suitable algorithms based on problem type
- **Result Interpretation**: Explains model performance and suggests improvements
- **Interactive Q&A**: Answer questions about any step or concept

## 🏗️ Project Structure

```
dataiku_lite/
├── app/
│   ├── data_analysis/          # Data profiling and exploration
│   ├── preprocessing/          # Preprocessing pipeline components
│   ├── modeling/              # Model training and evaluation
│   ├── ai_teacher/            # AI guidance and explanations
│   ├── visualization/         # Visualization components
│   └── utils/                 # Utility functions
├── tests/                     # Test files
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── config.py                 # Configuration settings
└── app.py                    # Main Streamlit application
```

## 🔧 Configuration

### Groq API Setup
1. Get your API key from [Groq Console](https://console.groq.com/)
2. Add it to your environment variables or config.py
3. The AI Teacher will use this for intelligent explanations

### Database Configuration (Optional)
- PostgreSQL for experiment tracking
- MongoDB for document storage
- Configure in config.py

## 📊 Supported Data Formats

- **CSV**: Comma-separated values
- **Excel**: .xlsx and .xls files
- **JSON**: JSON data files
- **Parquet**: Columnar data format
- **SQL Databases**: PostgreSQL, MySQL, SQLite

## 🎓 Learning Paths

### Beginner
1. Start with data upload and exploration
2. Follow AI-guided preprocessing steps
3. Use recommended models for training
4. Learn from AI explanations

### Intermediate
1. Customize preprocessing pipelines
2. Experiment with different algorithms
3. Tune hyperparameters
4. Compare model performance

### Advanced
1. Build custom preprocessing steps
2. Implement custom models
3. Advanced feature engineering
4. Model deployment considerations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with love for the data science community
- Inspired by platforms like Dataiku and AWS DataWrangler
- Powered by open-source libraries and tools

## 📞 Support

- Documentation: [Link to docs]
- Issues: [GitHub Issues]
- Discussions: [GitHub Discussions]

---

**Happy Learning! 🎉**

*Dataiku Lite - Making Data Science Accessible and Educational*
