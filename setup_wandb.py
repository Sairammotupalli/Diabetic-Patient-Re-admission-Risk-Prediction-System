#!/usr/bin/env python3
"""
Weights & Biases Setup Script
============================

This script helps you set up W&B for the diabetic readmission prediction project.
"""

import os
import subprocess
import sys

def install_wandb():
    """Install W&B package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ W&B installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install W&B")
        return False

def setup_api_key():
    """Guide user to set up W&B API key"""
    print("\n🔑 Setting up W&B API Key:")
    print("1. Go to https://wandb.ai/settings")
    print("2. Copy your API key")
    print("3. Set it as an environment variable:")
    print("   export WANDB_API_KEY=your_api_key_here")
    print("\nOr run this command:")
    print("wandb login")

def test_wandb():
    """Test W&B installation"""
    try:
        import wandb
        print("✅ W&B import successful")
        
        # Test basic functionality
        wandb.init(mode="disabled")
        print("✅ W&B initialization successful")
        wandb.finish()
        
        return True
    except ImportError:
        print("❌ W&B not installed")
        return False
    except Exception as e:
        print(f"❌ W&B test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🐳 Weights & Biases Setup for Diabetic Readmission Prediction")
    print("=" * 60)
    
    # Check if W&B is installed
    try:
        import wandb
        print("✅ W&B already installed")
    except ImportError:
        print("📦 Installing W&B...")
        if not install_wandb():
            print("❌ Setup failed. Please install manually:")
            print("pip install wandb")
            return
    
    # Test W&B
    print("\n🧪 Testing W&B installation...")
    if not test_wandb():
        print("❌ W&B test failed")
        return
    
    # Guide user to set up API key
    setup_api_key()
    
    print("\n✅ W&B setup completed!")
    print("\n📊 To use W&B with your project:")
    print("1. Set your API key: export WANDB_API_KEY=your_key")
    print("2. Run training: python main.py --train")
    print("3. View results: https://wandb.ai/your-username/diabetic-readmission-prediction")

if __name__ == "__main__":
    main() 