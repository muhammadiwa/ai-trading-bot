#!/usr/bin/env python3
"""
Test script untuk memvalidasi fungsi rounding dan trading
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.utils import round_coin_amount, validate_coin_amount

def test_rounding():
    """Test rounding functions"""
    print("Testing coin rounding functions...")
    
    test_cases = [
        0.001038545,  # Error case dari log
        1.123456789,  # 9 decimal places
        0.12345678,   # 8 decimal places
        0.1,          # 1 decimal place
        1.0,          # No decimal
        0.00000001,   # 8 decimal places minimum
        0.000000001,  # 9 decimal places
        1234.567890123456789  # Many decimal places
    ]
    
    for amount in test_cases:
        rounded = round_coin_amount(amount)
        is_valid_before = validate_coin_amount(amount)
        is_valid_after = validate_coin_amount(rounded)
        
        print(f"Original: {amount}")
        print(f"Rounded:  {rounded}")
        print(f"Valid before: {is_valid_before}")
        print(f"Valid after:  {is_valid_after}")
        print(f"String repr: '{str(rounded)}'")
        print("---")

if __name__ == "__main__":
    test_rounding()
