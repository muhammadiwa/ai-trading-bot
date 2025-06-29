"""
Risk management module for trading operations
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog

from core.database import get_db, User, Trade, Portfolio, UserSettings
from core.indodax_api import indodax_api

logger = structlog.get_logger(__name__)

class RiskManager:
    """Risk management for trading operations"""
    
    def __init__(self):
        self.max_daily_trades = 10
        self.max_position_size = 0.3  # 30% of portfolio
        self.stop_loss_percentage = 5.0  # 5%
        self.take_profit_percentage = 10.0  # 10%
    
    async def validate_trade(self, user: User, pair_id: str, trade_type: str, amount: float, price: float) -> Dict[str, Any]:
        """Validate if trade is within risk parameters"""
        try:
            validation_result = {
                "allowed": True,
                "warnings": [],
                "errors": [],
                "risk_score": 0.0
            }
            
            # Get user settings
            user_settings = await self._get_user_settings(user.id)
            
            # Check daily trade limit
            daily_trades = await self._get_daily_trades_count(user.id)
            max_daily = user_settings.get('max_daily_trades', self.max_daily_trades)
            
            if daily_trades >= max_daily:
                validation_result["allowed"] = False
                validation_result["errors"].append(f"Batas maksimal {max_daily} trade per hari tercapai")
            
            # Check trade amount vs balance
            balance_check = await self._check_balance_risk(user, trade_type, amount, price)
            if not balance_check["valid"]:
                validation_result["allowed"] = False
                validation_result["errors"].extend(balance_check["errors"])
            
            validation_result["warnings"].extend(balance_check["warnings"])
            
            # Check position size
            position_check = await self._check_position_size(user, pair_id, amount, price)
            if not position_check["valid"]:
                validation_result["warnings"].extend(position_check["warnings"])
            
            # Calculate risk score
            validation_result["risk_score"] = await self._calculate_risk_score(
                user, pair_id, trade_type, amount, price
            )
            
            return validation_result
            
        except Exception as e:
            logger.error("Failed to validate trade", error=str(e))
            return {
                "allowed": False,
                "warnings": [],
                "errors": ["Gagal memvalidasi trade"],
                "risk_score": 1.0
            }
    
    async def _get_user_settings(self, user_id: int) -> Dict[str, Any]:
        """Get user risk management settings"""
        try:
            db = get_db()
            try:
                settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                
                if not settings:
                    return {
                        "stop_loss_percentage": self.stop_loss_percentage,
                        "take_profit_percentage": self.take_profit_percentage,
                        "max_daily_trades": self.max_daily_trades
                    }
                
                return {
                    "stop_loss_percentage": settings.stop_loss_percentage,
                    "take_profit_percentage": settings.take_profit_percentage,
                    "max_daily_trades": self.max_daily_trades
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get user settings", user_id=user_id, error=str(e))
            return {}
    
    async def _get_daily_trades_count(self, user_id: int) -> int:
        """Get number of trades executed today"""
        try:
            db = get_db()
            try:
                today = datetime.now().date()
                tomorrow = today + timedelta(days=1)
                
                count = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.created_at >= today,
                    Trade.created_at < tomorrow
                ).count()
                
                return count
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get daily trades count", user_id=user_id, error=str(e))
            return 0
    
    async def _check_balance_risk(self, user: User, trade_type: str, amount: float, price: float) -> Dict[str, Any]:
        """Check balance-related risks"""
        try:
            result = {
                "valid": True,
                "warnings": [],
                "errors": []
            }
            
            # Get user balance
            from bot.utils import decrypt_api_key
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            user_api = indodax_api.__class__(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
            if trade_type == "buy":
                # Check IDR balance for buy orders
                idr_balance = float(balance_data.get('idr', 0))
                total_cost = amount
                
                if total_cost > idr_balance:
                    result["valid"] = False
                    result["errors"].append("Saldo IDR tidak mencukupi")
                elif total_cost > idr_balance * 0.8:
                    result["warnings"].append("Anda akan menggunakan lebih dari 80% saldo IDR")
                elif total_cost > idr_balance * 0.5:
                    result["warnings"].append("Anda akan menggunakan lebih dari 50% saldo IDR")
            
            else:  # sell
                # Check crypto balance for sell orders
                crypto_symbol = pair_id.split('_')[0]
                crypto_balance = float(balance_data.get(crypto_symbol, 0))
                crypto_amount = amount / price
                
                if crypto_amount > crypto_balance:
                    result["valid"] = False
                    result["errors"].append(f"Saldo {crypto_symbol.upper()} tidak mencukupi")
                elif crypto_amount > crypto_balance * 0.8:
                    result["warnings"].append(f"Anda akan menjual lebih dari 80% {crypto_symbol.upper()}")
            
            return result
            
        except Exception as e:
            logger.error("Failed to check balance risk", error=str(e))
            return {
                "valid": False,
                "warnings": [],
                "errors": ["Gagal memeriksa saldo"]
            }
    
    async def _check_position_size(self, user: User, pair_id: str, amount: float, price: float) -> Dict[str, Any]:
        """Check position size risk"""
        try:
            result = {
                "valid": True,
                "warnings": []
            }
            
            # Calculate total portfolio value
            total_portfolio_value = await self._get_portfolio_value(user)
            
            if total_portfolio_value <= 0:
                return result
            
            # Calculate position value
            position_value = amount if pair_id.endswith('_idr') else amount * price
            position_percentage = position_value / total_portfolio_value
            
            if position_percentage > self.max_position_size:
                result["warnings"].append(
                    f"Posisi ini akan menjadi {position_percentage*100:.1f}% dari portfolio "
                    f"(maks {self.max_position_size*100:.1f}%)"
                )
            
            return result
            
        except Exception as e:
            logger.error("Failed to check position size", error=str(e))
            return {"valid": True, "warnings": []}
    
    async def _get_portfolio_value(self, user: User) -> float:
        """Get total portfolio value in IDR"""
        try:
            from bot.utils import decrypt_api_key
            api_key = decrypt_api_key(user.indodax_api_key)
            secret_key = decrypt_api_key(user.indodax_secret_key)
            
            user_api = indodax_api.__class__(api_key, secret_key)
            balance_data = await user_api.get_balance()
            
            total_value = 0.0
            
            # Add IDR balance
            total_value += float(balance_data.get('idr', 0))
            
            # Add crypto balances (convert to IDR)
            for currency, balance in balance_data.items():
                if currency != 'idr' and float(balance) > 0:
                    try:
                        pair_id = f"{currency}_idr"
                        ticker_data = await indodax_api.get_ticker(pair_id)
                        
                        if ticker_data and 'ticker' in ticker_data:
                            price = float(ticker_data['ticker']['last'])
                            total_value += float(balance) * price
                    except:
                        continue
            
            return total_value
            
        except Exception as e:
            logger.error("Failed to get portfolio value", error=str(e))
            return 0.0
    
    async def _calculate_risk_score(self, user: User, pair_id: str, trade_type: str, amount: float, price: float) -> float:
        """Calculate risk score for the trade (0.0 = low risk, 1.0 = high risk)"""
        try:
            risk_factors = []
            
            # Portfolio concentration risk
            portfolio_value = await self._get_portfolio_value(user)
            if portfolio_value > 0:
                position_value = amount if pair_id.endswith('_idr') else amount * price
                concentration = position_value / portfolio_value
                risk_factors.append(min(concentration / 0.3, 1.0))  # Normalize to 30% max
            
            # Daily trade frequency risk
            daily_trades = await self._get_daily_trades_count(user.id)
            frequency_risk = min(daily_trades / self.max_daily_trades, 1.0)
            risk_factors.append(frequency_risk)
            
            # Market volatility risk (simplified)
            volatility_risk = await self._get_volatility_risk(pair_id)
            risk_factors.append(volatility_risk)
            
            # Calculate weighted average
            if risk_factors:
                return sum(risk_factors) / len(risk_factors)
            
            return 0.5  # Medium risk if no factors available
            
        except Exception as e:
            logger.error("Failed to calculate risk score", error=str(e))
            return 0.5
    
    async def _get_volatility_risk(self, pair_id: str) -> float:
        """Get volatility risk for a trading pair"""
        try:
            # Get recent price data
            ticker_data = await indodax_api.get_ticker(pair_id)
            
            if not ticker_data or 'ticker' not in ticker_data:
                return 0.5  # Medium risk if no data
            
            ticker = ticker_data['ticker']
            high = float(ticker.get('high', 0))
            low = float(ticker.get('low', 0))
            last = float(ticker.get('last', 0))
            
            if last > 0:
                # Calculate 24h volatility
                volatility = (high - low) / last
                
                # Normalize volatility to risk score
                if volatility > 0.2:  # >20% volatility = high risk
                    return 1.0
                elif volatility > 0.1:  # >10% volatility = medium-high risk
                    return 0.7
                elif volatility > 0.05:  # >5% volatility = medium risk
                    return 0.5
                else:  # <5% volatility = low risk
                    return 0.3
            
            return 0.5
            
        except Exception as e:
            logger.error("Failed to get volatility risk", pair_id=pair_id, error=str(e))
            return 0.5
    
    async def calculate_stop_loss_take_profit(self, entry_price: float, trade_type: str, user_settings: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stop loss and take profit levels"""
        try:
            stop_loss_pct = user_settings.get('stop_loss_percentage', self.stop_loss_percentage) / 100
            take_profit_pct = user_settings.get('take_profit_percentage', self.take_profit_percentage) / 100
            
            if trade_type == "buy":
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # sell
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
            
            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
            
        except Exception as e:
            logger.error("Failed to calculate stop loss/take profit", error=str(e))
            return {"stop_loss": 0.0, "take_profit": 0.0}
    
    async def check_stop_loss_take_profit(self, user_id: int) -> List[Dict[str, Any]]:
        """Check if any positions hit stop loss or take profit"""
        try:
            triggered_orders = []
            
            # This would check open positions against current prices
            # and return any that hit stop loss or take profit levels
            
            # Implementation would involve:
            # 1. Get user's open positions
            # 2. Get current prices
            # 3. Compare with stop loss/take profit levels
            # 4. Return triggered orders
            
            return triggered_orders
            
        except Exception as e:
            logger.error("Failed to check stop loss/take profit", user_id=user_id, error=str(e))
            return []

# Global risk manager instance
risk_manager = RiskManager()
