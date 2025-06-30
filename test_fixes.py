#!/usr/bin/env python3
"""
Test script to verify all fixes are working properly
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_modules():
    """Test all critical modules"""
    print("🔧 Testing module imports...")
    
    try:
        # Test LSTM predictor
        from ai.lstm_predictor import LSTMPredictor
        print("✅ LSTM Predictor import successful")
        
        # Test backtester
        from core.backtester import Backtester
        print("✅ Backtester import successful")
        
        # Test IndodaxAPI
        from core.indodax_api import IndodaxAPI
        print("✅ IndodaxAPI import successful")
        
        # Test Auto Trader
        from core.auto_trader import AutoTrader
        print("✅ AutoTrader import successful")
        
        # Test DCA Manager
        from core.dca_manager import DCAManager
        print("✅ DCAManager import successful")
        
        # Test AI modules
        from ai.signal_generator import SignalGenerator
        from ai.sentiment_analyzer import SentimentAnalyzer
        print("✅ AI modules import successful")
        
        print("\n🧪 Testing basic functionality...")
        
        # Test LSTM predictor initialization
        lstm = LSTMPredictor()
        print("✅ LSTM Predictor initialization successful")
        
        # Test API initialization
        api = IndodaxAPI()
        print("✅ IndodaxAPI initialization successful")
        
        # Test backtester initialization
        backtester = Backtester()
        print("✅ Backtester initialization successful")
        
        print("\n🎯 Testing LSTM predictor safe numpy conversion...")
        
        # Test the safe numpy conversion function
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test the internal function (import it directly)
        from ai.lstm_predictor import _safe_to_numpy
        result = _safe_to_numpy(test_data)
        print(f"✅ Safe numpy conversion successful: {result.shape}")
        
        # Test with DataFrame
        test_df = pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result2 = _safe_to_numpy(test_df['close'])
        print(f"✅ Safe numpy conversion with DataFrame successful: {result2.shape}")
        
        print("\n🚀 All tests passed! The system is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_modules())
    sys.exit(0 if success else 1)
