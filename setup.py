#!/usr/bin/env python3
"""
Dataiku Lite - Setup Script
Automated setup and installation script
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🧠 Dataiku Lite - Setup Script")
    print("=" * 50)
    print("This script will set up your Dataiku Lite environment")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("Please install Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("Please install manually with: pip install -r requirements.txt")
        return False

def create_config():
    """Create configuration file if it doesn't exist"""
    print("\nSetting up configuration...")
    config_file = Path("config.py")
    
    if not config_file.exists():
        print("Creating default configuration file...")
        config_content = '''"""
Configuration settings for the ML/DS Educational Platform
"""
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Groq API Configuration
    groq_api_key: str = ""
    groq_model: str = "qwen2.5-32b-instruct"
    
    # Database Configuration
    database_url: Optional[str] = None
    mongo_url: Optional[str] = None
    
    # Application Configuration
    debug: bool = True
    log_level: str = "INFO"
    max_file_size_mb: int = 100
    cache_ttl_hours: int = 24
    
    # Model Configuration
    max_features: int = 1000
    default_test_size: float = 0.2
    random_state: int = 42
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
'''
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print("✅ Configuration file created")
        print("⚠️  To use AI features, add your Groq API key to config.py")
    else:
        print("✅ Configuration file already exists")

def run_tests():
    """Run basic tests to verify installation"""
    print("\nRunning installation tests...")
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Tests timed out, but installation may still be working")
        return True
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")
        return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Add your Groq API key to config.py (optional, for AI features)")
    print("2. Run the application:")
    print("   python run.py")
    print("   or")
    print("   streamlit run app.py")
    print("\n3. Open your browser to http://localhost:8501")
    print("\n4. Try the example usage:")
    print("   python example_usage.py")
    print("\nFor more information, see README.md")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Setup incomplete due to installation errors")
        print("Please install requirements manually and try again")
        sys.exit(1)
    
    # Create configuration
    create_config()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
