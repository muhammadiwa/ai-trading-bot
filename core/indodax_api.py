"""
Indodax API client for trading operations
"""
import hashlib
import hmac
import time
import requests
import json
from typing import Dict, List, Optional, Any
from config.settings import settings
import structlog

logger = structlog.get_logger(__name__)

class IndodaxAPI:
    """Indodax API client for both public and private endpoints"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or settings.indodax_api_key
        self.secret_key = secret_key or settings.indodax_secret_key
        self.base_url = settings.indodax_base_url
        self.public_url = f"{self.base_url}/api"
        self.tapi_url = f"{self.base_url}/tapi"
        
    def _generate_signature(self, params: str) -> str:
        """Generate HMAC-SHA512 signature for private API calls"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds (as required by Indodax)"""
        return int(time.time() * 1000)
    
    # Public API Methods
    
    async def get_server_time(self) -> Dict[str, Any]:
        """Get server time from Indodax"""
        try:
            response = requests.get(f"{self.public_url}/server_time")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get server time", error=str(e))
            raise
    
    async def get_pairs(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs"""
        try:
            response = requests.get(f"{self.public_url}/pairs")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get trading pairs", error=str(e))
            raise
    
    async def get_price_increments(self) -> Dict[str, Any]:
        """Get price increments for all pairs"""
        try:
            response = requests.get(f"{self.public_url}/price_increments")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get price increments", error=str(e))
            raise
    
    async def get_summaries(self) -> Dict[str, Any]:
        """Get summary information for all pairs"""
        try:
            response = requests.get(f"{self.public_url}/summaries")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get summaries", error=str(e))
            raise
    
    async def get_ticker(self, pair_id: str = "btcidr") -> Dict[str, Any]:
        """Get ticker for specific pair"""
        try:
            response = requests.get(f"{self.public_url}/ticker/{pair_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get ticker", pair_id=pair_id, error=str(e))
            raise
    
    async def get_ticker_all(self) -> Dict[str, Any]:
        """Get all tickers"""
        try:
            response = requests.get(f"{self.public_url}/ticker_all")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get all tickers", error=str(e))
            raise
    
    async def get_trades(self, pair_id: str = "btcidr") -> List[Dict[str, Any]]:
        """Get recent trades for specific pair"""
        try:
            response = requests.get(f"{self.public_url}/trades/{pair_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get trades", pair_id=pair_id, error=str(e))
            raise
    
    async def get_depth(self, pair_id: str = "btcidr") -> Dict[str, Any]:
        """Get order book depth for specific pair"""
        try:
            response = requests.get(f"{self.public_url}/depth/{pair_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get depth", pair_id=pair_id, error=str(e))
            raise
    
    async def get_ohlc_history(self, pair_id: str, from_time: int, to_time: int, timeframe: str = "15") -> List[Dict[str, Any]]:
        """Get OHLC history data"""
        try:
            url = f"{self.base_url}/tradingview/history_v2"
            params = {
                "symbol": pair_id.upper(),
                "from": from_time,
                "to": to_time,
                "tf": timeframe
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get OHLC history", pair_id=pair_id, error=str(e))
            raise
    
    # Private API Methods
    
    async def _private_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request to private API"""
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key are required for private API calls")
        
        if params is None:
            params = {}
        
        # Add required parameters
        params["method"] = method
        params["timestamp"] = self._get_timestamp()
        params["recvWindow"] = 5000
        
        # Create query string for signature (without the signature itself)
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Generate signature
        signature = self._generate_signature(query_string)
        
        # Prepare headers
        headers = {
            "Key": self.api_key,
            "Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        try:
            # Use proper async request handling
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(self.tapi_url, data=params, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    # Check for API errors
                    if result.get("success") == 0:
                        error_msg = result.get("error", "Unknown API error")
                        error_code = result.get("error_code", "unknown")
                        logger.error("Indodax API error", method=method, error=error_msg, error_code=error_code)
                        raise Exception(f"Indodax API Error [{error_code}]: {error_msg}")
                    
                    logger.info("API request successful", method=method)
                    return result
            
        except aiohttp.ClientError as e:
            logger.error("HTTP request failed", method=method, error=str(e))
            raise Exception(f"Connection error: {str(e)}")
        except Exception as e:
            logger.error("Private API request failed", method=method, error=str(e))
            raise
    
    async def get_info(self) -> Dict[str, Any]:
        """Get account information"""
        return await self._private_request("getInfo")
    
    async def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        info = await self.get_info()
        return info.get("return", {}).get("balance", {})
    
    async def trade(self, pair: str, type: str, price: float, amount: float) -> Dict[str, Any]:
        """Execute a trade order"""
        params = {
            "pair": pair,
            "type": type,  # buy or sell
            "price": str(price),
            "amount": str(amount)
        }
        return await self._private_request("trade", params)
    
    async def cancel_order(self, order_id: str, pair: str, type: str) -> Dict[str, Any]:
        """Cancel an order"""
        params = {
            "order_id": order_id,
            "pair": pair,
            "type": type
        }
        return await self._private_request("cancelOrder", params)
    
    async def get_open_orders(self, pair: str = None) -> Dict[str, Any]:
        """Get open orders"""
        params = {}
        if pair:
            params["pair"] = pair
        return await self._private_request("openOrders", params)
    
    async def get_order_history(self, pair: str = None, count: int = 1000) -> Dict[str, Any]:
        """Get order history"""
        params = {"count": count}
        if pair:
            params["pair"] = pair
        return await self._private_request("orderHistory", params)
    
    async def get_trade_history(self, pair: str = None, count: int = 1000) -> Dict[str, Any]:
        """Get trade history"""
        params = {"count": count}
        if pair:
            params["pair"] = pair
        return await self._private_request("tradeHistory", params)

# Global API instance
indodax_api = IndodaxAPI()
