"""
Setup script for Omni-Assistant Discord Bot.
Helps with initial configuration and dependency installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ .env file created")
        print("⚠️  Please edit .env file with your Discord bot token and settings")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ .env.example not found")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["data", "logs", "data/vector_db"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def check_lm_studio():
    """Check if LM Studio is accessible."""
    print("🔍 Checking LM Studio connection...")
    try:
        import requests
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running and accessible")
            return True
        else:
            print("⚠️  LM Studio responded but may not be properly configured")
            return False
    except requests.exceptions.RequestException:
        print("⚠️  LM Studio is not running or not accessible at http://127.0.0.1:1234")
        print("   Please start LM Studio and load a model before running the bot")
        return False
    except ImportError:
        print("⚠️  Cannot check LM Studio (requests not installed yet)")
        return False

def main():
    """Main setup function."""
    print("""
    ╔═══════════════════════════════════════╗
    ║        Omni-Assistant Setup           ║
    ║         Discord AI Bot v1.0           ║
    ╚═══════════════════════════════════════╝
    """)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Set up environment
    if success and not setup_environment():
        success = False
    
    # Create directories
    if success:
        create_directories()
    
    # Check LM Studio (optional)
    check_lm_studio()
    
    print("\n" + "="*50)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your Discord bot token")
        print("2. Make sure LM Studio is running with a loaded model")
        print("3. Run the bot: python src/main.py")
    else:
        print("❌ Setup encountered errors. Please check the messages above.")
    
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()
