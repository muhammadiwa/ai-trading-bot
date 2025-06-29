#!/usr/bin/env python3
"""
Debug script untuk testing Indodax API credentials
"""

import asyncio
import sys
import os
import hashlib
import hmac
import time
import json

# Add project root to path
sys.path.append('/var/www/html/telebot-ai')

from core.indodax_api import IndodaxAPI
from config.settings import settings
import structlog

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def manual_signature_test():
    """Test signature generation manually like the docs example"""
    
    # Test values from documentation
    secret_key = settings.indodax_secret_key
    api_key = settings.indodax_api_key
    
    print(f"API Key: {api_key}")
    print(f"Secret Key: {secret_key[:20]}...")
    
    # Current timestamp
    timestamp = int(time.time() * 1000)
    recv_window = 5000
    
    # Create request body exactly like docs
    request_body = f"method=getInfo&recvWindow={recv_window}&timestamp={timestamp}"
    print(f"Request Body: {request_body}")
    
    # Generate signature
    signature = hmac.new(
        secret_key.encode('utf-8'),
        request_body.encode('utf-8'),
        hashlib.sha512
    ).hexdigest()
    
    print(f"Generated Signature: {signature}")
    
    return {
        'api_key': api_key,
        'secret_key': secret_key,
        'request_body': request_body,
        'signature': signature,
        'timestamp': timestamp,
        'recv_window': recv_window
    }

async def test_api_credentials():
    """Test API credentials with detailed logging"""
    
    print("=== INDODAX API CREDENTIALS TEST ===")
    print()
    
    # Manual signature test first
    manual_data = manual_signature_test()
    print()
    
    # Test with our API client
    print("=== TESTING WITH API CLIENT ===")
    
    try:
        api = IndodaxAPI()
        
        print(f"Using API Key: {api.api_key}")
        print(f"Using Secret Key: {api.secret_key[:20]}...")
        print(f"Base URL: {api.base_url}")
        print(f"TAPI URL: {api.tapi_url}")
        
        print("\nTesting getInfo method...")
        result = await api.get_info()
        
        print("✅ API credentials are valid!")
        print(f"Server time: {result.get('return', {}).get('server_time')}")
        
        # Test balance
        balance = await api.get_balance()
        print(f"Account balance keys: {list(balance.keys())}")
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        logger.error("API test failed", error=str(e), exc_info=True)
        
        # Additional debugging
        print("\n=== DEBUGGING INFO ===")
        print(f"Error type: {type(e).__name__}")
        
        if "bad_sign" in str(e).lower():
            print("❌ Signature error detected!")
            print("This could be caused by:")
            print("1. Wrong API secret key")
            print("2. Incorrect signature generation")
            print("3. Wrong timestamp format")
            print("4. Wrong parameter ordering")
            
        elif "invalid_credentials" in str(e).lower():
            print("❌ Invalid credentials error!")
            print("This could be caused by:")
            print("1. Wrong API key")
            print("2. API key permissions insufficient")
            print("3. API key expired or revoked")

async def test_public_api():
    """Test public API endpoints (no auth required)"""
    print("\n=== TESTING PUBLIC API ===")
    
    try:
        api = IndodaxAPI()
        
        # Test server time
        server_time = await api.get_server_time()
        print(f"✅ Server time: {server_time}")
        
        # Test ticker
        ticker = await api.get_ticker("btcidr")
        print(f"✅ BTC/IDR ticker: {ticker}")
        
    except Exception as e:
        print(f"❌ Public API test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_public_api())
    asyncio.run(test_api_credentials())
