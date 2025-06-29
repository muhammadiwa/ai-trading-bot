#!/usr/bin/env python3
"""
Test script to simulate sell flow steps
"""
import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.indodax_api import IndodaxAPI
from core.database import get_db
from core.database.models import User
from config.settings import Settings
from bot.utils import decrypt_api_key

settings = Settings()

async def test_sell_flow():
    """Test the sell flow with actual Indodax API"""
    try:
        print("🔍 Testing sell flow...")
        
        # Get first user from database (assumes we have a registered user)
        db = get_db()
        try:
            user = db.query(User).filter(User.indodax_api_key.isnot(None)).first()
            if not user:
                print("❌ No registered user found in database. Please register first using /daftar command.")
                return
            
            print(f"✅ Found registered user: {user.username or user.first_name}")
            
            # Decrypt API credentials
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            # Create API client
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Test 1: Get balance
            print("\n📊 Testing balance retrieval...")
            balance_data = await user_api.get_balance()
            print("✅ Balance data retrieved successfully")
            
            # Show IDR and some crypto balances
            idr_balance = float(balance_data.get('idr', 0))
            btc_balance = float(balance_data.get('btc', 0))
            eth_balance = float(balance_data.get('eth', 0))
            
            print(f"   💰 IDR Balance: {idr_balance:,.2f} IDR")
            print(f"   🟡 BTC Balance: {btc_balance:.8f} BTC")
            print(f"   🔵 ETH Balance: {eth_balance:.8f} ETH")
            
            # Test 2: Get ticker for price calculation
            print("\n📈 Testing ticker retrieval...")
            ticker = await user_api.get_ticker("btcidr")
            
            if "ticker" in ticker:
                buy_price = float(ticker["ticker"]["buy"])
                sell_price = float(ticker["ticker"]["sell"])
                print(f"✅ BTC/IDR ticker retrieved successfully")
                print(f"   Buy price: {buy_price:,.0f} IDR")
                print(f"   Sell price: {sell_price:,.0f} IDR")
                
                # Test 3: Calculate sell scenarios
                print("\n🧮 Testing sell calculations...")
                
                if btc_balance > 0:
                    # Test different percentages
                    percentages = [25, 50, 75, 100]
                    for pct in percentages:
                        sell_amount = btc_balance * pct / 100
                        estimated_idr = sell_amount * buy_price  # Use buy price for selling
                        
                        print(f"   {pct}% of BTC: {sell_amount:.8f} BTC = ~{estimated_idr:,.0f} IDR")
                        
                        # Check minimum order requirement
                        if estimated_idr >= 10000:
                            print(f"      ✅ Above minimum order (10,000 IDR)")
                        else:
                            print(f"      ❌ Below minimum order (need 10,000 IDR, got {estimated_idr:,.0f} IDR)")
                else:
                    print("   ⚠️  No BTC balance available for sell testing")
                
                # Test 4: Validate sell order (dry run - don't actually place order)
                print("\n🔍 Testing sell order validation...")
                
                # Simulate a small sell order validation
                test_sell_amount = 0.001  # 0.001 BTC
                test_idr_value = test_sell_amount * buy_price
                
                print(f"   Test sell: {test_sell_amount:.8f} BTC")
                print(f"   Estimated value: {test_idr_value:,.0f} IDR")
                
                if test_idr_value >= 10000:
                    print(f"   ✅ Test order would pass minimum requirement")
                else:
                    print(f"   ❌ Test order below minimum (need 10,000 IDR)")
                
                if btc_balance >= test_sell_amount:
                    print(f"   ✅ Sufficient BTC balance for test order")
                else:
                    print(f"   ❌ Insufficient BTC balance (have {btc_balance:.8f}, need {test_sell_amount:.8f})")
                
            else:
                print("❌ Failed to get ticker data")
            
            print("\n🎉 Sell flow test completed successfully!")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_sell_flow())
