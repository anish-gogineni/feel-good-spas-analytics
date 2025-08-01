#!/usr/bin/env python3
"""
Startup script for Feel Good Spas applications
"""

import os
import subprocess
import sys
import time

def set_api_key():
    """Set the OpenAI API key in environment"""
    # Replace with your actual API key before running
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
    os.environ['OPENAI_API_KEY'] = api_key
    print("✅ OpenAI API key configured")

def test_setup():
    """Test that everything is working"""
    print("🔍 Testing setup...")
    
    # Test data loading
    try:
        import pandas as pd
        df = pd.read_csv('data/processed_calls_enriched.csv')
        print(f"✅ Data loaded: {len(df)} records")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test OpenAI API
    try:
        import openai
        openai.api_key = os.environ['OPENAI_API_KEY']
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input="test"
        )
        print(f"✅ OpenAI API working")
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False
    
    return True

def clean_cache():
    """Clean Streamlit cache and embedding cache"""
    print("🧹 Cleaning caches...")
    
    # Remove embedding cache
    try:
        import os
        if os.path.exists('data/call_embeddings.pkl'):
            os.remove('data/call_embeddings.pkl')
        if os.path.exists('data/call_embeddings_indices.pkl'):
            os.remove('data/call_embeddings_indices.pkl')
        print("✅ Embedding cache cleared")
    except Exception as e:
        print(f"⚠️ Cache cleaning warning: {e}")

def start_dashboard():
    """Start the dashboard app"""
    print("\n🚀 Starting Dashboard App...")
    print("Dashboard will be available at: http://localhost:8501")
    
    env = os.environ.copy()
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app/dashboard.py",
        "--server.port", "8501"
    ], env=env)

def start_chat():
    """Start the chat app"""
    print("\n🤖 Starting Chat App...")
    print("Chat will be available at: http://localhost:8502")
    
    env = os.environ.copy()
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app/chat.py",
        "--server.port", "8502"
    ], env=env)

def main():
    """Main function"""
    print("🌟 Feel Good Spas - Application Launcher")
    print("========================================")
    
    # Setup
    set_api_key()
    
    if not test_setup():
        print("❌ Setup test failed. Please check the errors above.")
        return
    
    clean_cache()
    
    # Ask user which app to start
    print("\n📱 Which app would you like to start?")
    print("1. 📊 Dashboard (Analytics & Visualizations)")
    print("2. 🤖 Chat Assistant (AI-powered Q&A)")
    print("3. 🔧 Run diagnostic tests only")
    
    try:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            start_dashboard()
        elif choice == "2":
            start_chat()
        elif choice == "3":
            print("✅ All diagnostic tests passed!")
            print("\nTo start apps manually:")
            print("Dashboard: streamlit run app/dashboard.py --server.port 8501")
            print("Chat: streamlit run app/chat.py --server.port 8502")
        else:
            print("❌ Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 