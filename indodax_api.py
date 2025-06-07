import os
import requests
import hashlib
import hmac
import time
import json
from typing import Dict, List, Optional
from datetime import datetime

class IndodaxAPI:
    """
    Kelas untuk berinteraksi dengan API Indodax
    """
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://indodax.com"
        
    def _generate_signature(self, method: str, path: str, body: str = "") -> str:
        """Generate signature untuk autentikasi"""
        timestamp = str(int(time.time() * 1000))
        string_to_sign = f"{method}\n{path}\n{body}\n{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature, timestamp
    
    def get_ticker(self, pair: str = "btcidr") -> Dict:
        """Mendapatkan harga ticker"""
        try:
            url = f"{self.base_url}/api/{pair}/ticker"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            print(f"Error getting ticker: {e}")
            return {}
    
    def get_orderbook(self, pair: str = "btcidr") -> Dict:
        """Mendapatkan order book"""
        try:
            url = f"{self.base_url}/api/{pair}/depth"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            print(f"Error getting orderbook: {e}")
            return {}
    
    def get_trades(self, pair: str = "btcidr") -> List[Dict]:
        """Mendapatkan riwayat transaksi"""
        try:
            url = f"{self.base_url}/api/{pair}/trades"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            print(f"Error getting trades: {e}")
            return []
    
    def get_account_info(self) -> Dict:
        """Mendapatkan informasi akun"""
        try:
            path = "/api/getInfo"
            method = "POST"
            body = ""
            
            signature, timestamp = self._generate_signature(method, path, body)
            
            headers = {
                "Key": self.api_key,
                "Sign": signature,
                "timestamp": timestamp,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body)
            return response.json()
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def place_buy_order(self, pair: str, amount: float, price: float) -> Dict:
        """Menempatkan order beli"""
        try:
            path = "/api/trade"
            method = "POST"
            
            data = {
                "pair": pair,
                "type": "buy",
                "price": str(price),
                pair.split("idr")[0]: str(amount)
            }
            
            body = "&".join([f"{k}={v}" for k, v in data.items()])
            signature, timestamp = self._generate_signature(method, path, body)
            
            headers = {
                "Key": self.api_key,
                "Sign": signature,
                "timestamp": timestamp,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body)
            return response.json()
        except Exception as e:
            print(f"Error placing buy order: {e}")
            return {}
    
    def place_sell_order(self, pair: str, amount: float, price: float) -> Dict:
        """Menempatkan order jual"""
        try:
            path = "/api/trade"
            method = "POST"
            
            data = {
                "pair": pair,
                "type": "sell",
                "price": str(price),
                pair.split("idr")[0]: str(amount)
            }
            
            body = "&".join([f"{k}={v}" for k, v in data.items()])
            signature, timestamp = self._generate_signature(method, path, body)
            
            headers = {
                "Key": self.api_key,
                "Sign": signature,
                "timestamp": timestamp,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body)
            return response.json()
        except Exception as e:
            print(f"Error placing sell order: {e}")
            return {}
    
    def get_open_orders(self, pair: str = "btcidr") -> Dict:
        """Mendapatkan order yang masih aktif"""
        try:
            path = "/api/openOrders"
            method = "POST"
            
            data = {"pair": pair}
            body = "&".join([f"{k}={v}" for k, v in data.items()])
            signature, timestamp = self._generate_signature(method, path, body)
            
            headers = {
                "Key": self.api_key,
                "Sign": signature,
                "timestamp": timestamp,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body)
            return response.json()
        except Exception as e:
            print(f"Error getting open orders: {e}")
            return {}
    
    def cancel_order(self, pair: str, order_id: str, order_type: str) -> Dict:
        """Membatalkan order"""
        try:
            path = "/api/cancelOrder"
            method = "POST"
            
            data = {
                "pair": pair,
                "order_id": order_id,
                "type": order_type
            }
            
            body = "&".join([f"{k}={v}" for k, v in data.items()])
            signature, timestamp = self._generate_signature(method, path, body)
            
            headers = {
                "Key": self.api_key,
                "Sign": signature,
                "timestamp": timestamp,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            url = f"{self.base_url}{path}"
            response = requests.post(url, headers=headers, data=body)
            return response.json()
        except Exception as e:
            print(f"Error canceling order: {e}")
            return {}
