#!/usr/bin/env python3
"""
Dataiku Lite - Run Script
Simple script to start the application with proper setup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import pandas
        import numpy
        import streamlit
        import plotly
        import sklearn
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def check_config():
    """Check if configuration is set up"""
    config_file = Path("config.py")
    if not config_file.exists():
        print("⚠️  Configuration file not found. Using default settings.")
        print("To use AI features, add your Groq API key to config.py")
        return False
    return True

def main():
    """Main function to run the application"""
    print("🧠 Dataiku Lite - Starting Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check configuration
    check_config()
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    
    print("✅ Requirements check passed")
    print("🚀 Starting Streamlit application...")
    print("\nThe application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
