"""
Test script to verify Omni-Assistant installation and configuration.
Run this script to check if everything is set up correctly.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    if sys.version_info >= (3, 8):
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
        return True
    else:
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is too old (need 3.8+)")
        return False

def test_dependencies():
    """Test if required dependencies are installed."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        ('discord.py', 'discord'),
        ('crewai', 'crewai'),
        ('langchain', 'langchain'),
        ('chromadb', 'chromadb'),
        ('sentence_transformers', 'sentence_transformers'),
        ('requests', 'requests'),
        ('beautifulsoup4', 'bs4'),
        ('aiohttp', 'aiohttp'),
        ('python-dotenv', 'dotenv'),
        ('pydantic', 'pydantic'),
        ('crewai-tools', 'crewai_tools')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_file_structure():
    """Test if all required files and directories exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'src/main.py',
        'src/config.py',
        'src/llm_integration.py',
        'src/memory_manager.py',
        'src/personalities.py',
        'src/crew_orchestrator.py',
        'src/discord_handlers.py',
        'src/agents/base_agent.py',
        'src/agents/research_agent.py',
        'src/agents/coding_agent.py',
        'src/tools/web_browser_tool.py',
        'src/tools/coding_tool.py',
        'src/tools/vector_db_tool.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'src',
        'src/agents',
        'src/tools',
        'data',
        'logs'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_good = False
    
    return all_good

def test_configuration():
    """Test configuration files."""
    print("\n⚙️  Testing configuration...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_example.exists():
        print("❌ .env.example file missing")
        return False
    else:
        print("✅ .env.example exists")
    
    if not env_file.exists():
        print("⚠️  .env file not found - you'll need to create it")
        return False
    else:
        print("✅ .env file exists")
        
        # Check if token is configured
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'your_discord_bot_token_here' in content:
                    print("⚠️  Discord bot token not configured in .env")
                    return False
                elif 'DISCORD_BOT_TOKEN=' in content:
                    print("✅ Discord bot token appears to be configured")
                    return True
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
            return False
    
    return True

def test_imports():
    """Test if the main modules can be imported."""
    print("\n🔧 Testing module imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path('src')))
    
    modules_to_test = [
        'config',
        'personalities',
        'memory_manager',
        'llm_integration',
        'crew_orchestrator',
        'discord_handlers'
    ]
    
    all_good = True
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            all_good = False
    
    return all_good

def test_lm_studio_connection():
    """Test LM Studio connection."""
    print("\n🤖 Testing LM Studio connection...")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running and accessible")
            models = response.json()
            if models.get('data'):
                print(f"✅ Found {len(models['data'])} model(s) loaded")
            else:
                print("⚠️  No models loaded in LM Studio")
            return True
        else:
            print(f"⚠️  LM Studio responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("⚠️  LM Studio is not running or not accessible")
        print("   Start LM Studio and load a model before running the bot")
        return False
    except Exception as e:
        print(f"❌ Error testing LM Studio: {e}")
        return False

def main():
    """Run all tests."""
    print("""
    ╔═══════════════════════════════════════╗
    ║      Omni-Assistant Test Suite        ║
    ║         Installation Verification     ║
    ╚═══════════════════════════════════════╝
    """)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("Configuration", test_configuration),
        ("Module Imports", test_imports),
        ("LM Studio Connection", test_lm_studio_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation looks good.")
        print("You can now run the bot with: python src/main.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        print("Refer to README.md for setup instructions.")

if __name__ == "__main__":
    main()
