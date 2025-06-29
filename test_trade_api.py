#!/usr/bin/env python3
"""
Test script untuk debugging trade API calls
"""
import asyncio
import os
from core.indodax_api import IndodaxAPI
from config.settings import settings

async def test_trade_api():
    # Baca API credentials dari environment
    api_key = os.getenv("INDODAX_API_KEY")
    secret_key = os.getenv("INDODAX_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ INDODAX_API_KEY dan INDODAX_SECRET_KEY harus diset di environment")
        return
    
    # Inisialisasi API
    api = IndodaxAPI(api_key, secret_key)
    
    try:
        # Test 1: Get ticker untuk harga
        print("🔍 Testing get ticker for BNB...")
        ticker = await api.get_ticker("bnbidr")
        print(f"✅ BNB Ticker: {ticker}")
        
        if "ticker" in ticker:
            price = float(ticker["ticker"]["sell"])
            print(f"💰 Current BNB sell price: {price:,.0f} IDR")
            
            # Test 2: Calculate trade parameters
            idr_amount = 12000  # 12,000 IDR
            coin_amount = idr_amount / price
            
            print(f"📊 Trade calculation:")
            print(f"   IDR Amount: {idr_amount:,.0f}")
            print(f"   BNB Price: {price:,.0f}")
            print(f"   BNB Quantity: {coin_amount:.8f}")
            print(f"   Total Value: {coin_amount * price:,.0f} IDR")
            
            # Test 3: Show what parameters would be sent
            print(f"\n📤 Buy order parameters would be:")
            print(f"   pair: bnb_idr")
            print(f"   type: buy")
            print(f"   price: {price}")
            print(f"   idr: {int(idr_amount)}")
            
            print(f"\n📤 Sell order parameters would be:")
            print(f"   pair: bnb_idr")
            print(f"   type: sell")
            print(f"   price: {price}")
            print(f"   bnb: {coin_amount:.8f}")
            
            # Note: Tidak melakukan trade actual untuk test
            print(f"\n⚠️  Tidak melakukan trade actual - hanya test parameter")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_trade_api())
