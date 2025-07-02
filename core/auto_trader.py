"""
Auto-trading system that executes trades based on AI signals
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

from core.database import get_db, User, AISignal, Trade, UserSettings
from core.indodax_api import IndodaxAPI
from core.risk_manager import RiskManager
from ai.signal_generator import SignalGenerator
from bot.utils import decrypt_api_key, encrypt_api_key
from bot.telegram_bot import TelegramBot

logger = structlog.get_logger(__name__)

class AutoTrader:
    """Automated trading system based on AI signals"""
    
    def __init__(self, telegram_bot: Optional[TelegramBot] = None):
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.telegram_bot = telegram_bot
        self.min_signal_confidence = 0.7  # Minimum confidence for auto-trading
        self.max_trades_per_day = 5  # Maximum auto-trades per user per day
    
    async def process_auto_trades(self):
        """Process auto-trades for all users with auto-trading enabled"""
        try:
            logger.info("Starting auto-trade processing")
            
            db = get_db()
            try:
                # Get users with auto-trading enabled
                auto_trading_users = db.query(User).filter(
                    User.auto_trading_enabled == True,
                    User.is_active == True,
                    User.indodax_api_key.isnot(None)
                ).all()
                
                if not auto_trading_users:
                    logger.info("No users with auto-trading enabled")
                    return
                
                logger.info("Processing auto-trades for users", count=len(auto_trading_users))
                
                # Process each user
                for user in auto_trading_users:
                    try:
                        await self._process_user_auto_trades(user)
                    except Exception as e:
                        logger.error("Failed to process auto-trades for user", 
                                   user_id=user.id, error=str(e))
                        continue
                
                logger.info("Auto-trade processing completed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to process auto-trades", error=str(e))
    
    async def _process_user_auto_trades(self, user: User):
        """Process auto-trades for a specific user"""
        try:
            # Cast user.id to int to fix type checking
            user_id = int(user.id)  # type: ignore
            
            # Check daily trade limit
            daily_trades = await self._get_daily_auto_trades_count(user_id)
            if daily_trades >= self.max_trades_per_day:
                logger.info("Daily auto-trade limit reached", user_id=user_id)
                return
            
            # Get user settings
            user_settings = await self._get_user_settings(user_id)
            
            # Get active trading pairs
            trading_pairs = await self._get_user_trading_pairs(user_id)
            
            for pair_id in trading_pairs:
                try:
                    await self._check_and_execute_auto_trade(user, pair_id, user_settings)
                except Exception as e:
                    logger.error("Failed to check auto-trade for pair", 
                               user_id=user_id, pair_id=pair_id, error=str(e))
                    continue
            
        except Exception as e:
            # Use user.id for error logging since user_id might not be defined yet
            logger.error("Failed to process user auto-trades", user_id=int(user.id), error=str(e))  # type: ignore
    
    async def _check_and_execute_auto_trade(self, user: User, pair_id: str, user_settings: Dict[str, Any]):
        """Check for trading signals and execute auto-trade if conditions are met"""
        try:
            # Cast user.id to int for function calls
            user_id = int(user.id)  # type: ignore
            telegram_id = int(user.telegram_id)  # type: ignore
            
            # Generate or get latest signal
            signal = await self.signal_generator.generate_signal(pair_id)
            
            if not signal:
                return
            
            # Extract signal values for comparison
            signal_confidence = getattr(signal, 'confidence', 0.0)
            signal_type = getattr(signal, 'signal_type', 'hold')
            
            # Check signal confidence
            if signal_confidence < self.min_signal_confidence:
                logger.debug("Signal confidence too low for auto-trading", 
                           pair_id=pair_id, confidence=signal_confidence)
                return
            
            # Check if signal is actionable (buy/sell, not hold)
            if signal_type == "hold":
                return
            
            # Check if we already have a recent trade for this signal
            if await self._has_recent_signal_trade(user_id, pair_id, signal_type):
                logger.debug("Recent trade already exists for this signal", 
                           user_id=user_id, pair_id=pair_id, signal_type=signal_type)
                return
            
            # Calculate trade amount based on user settings
            trade_amount = await self._calculate_auto_trade_amount(user, pair_id, user_settings)
            
            if trade_amount <= 0:
                logger.warning("Invalid trade amount calculated", 
                             user_id=user_id, pair_id=pair_id, amount=trade_amount)
                return
            
            # Execute auto-trade
            await self._execute_auto_trade(user, pair_id, signal, trade_amount)
            
        except Exception as e:
            logger.error("Failed to check and execute auto-trade", 
                       user_id=int(user.id), pair_id=pair_id, error=str(e))  # type: ignore
    
    async def _execute_auto_trade(self, user: User, pair_id: str, signal: AISignal, amount: float):
        """Execute automatic trade based on signal"""
        try:
            # Cast user attributes to proper types
            user_id = int(user.id)  # type: ignore
            telegram_id = int(user.telegram_id)  # type: ignore
            
            # Extract signal values
            signal_type = getattr(signal, 'signal_type', 'hold')
            
            # Get current market price
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Get ticker for current price
            ticker_pair = pair_id.replace("_", "") if "_" in pair_id else f"{pair_id}idr"
            ticker = await user_api.get_ticker(ticker_pair)
            
            if not ticker or "ticker" not in ticker:
                logger.error("Failed to get ticker data", pair_id=pair_id)
                return
            
            ticker_data = ticker["ticker"]
            
            if signal_type == "buy":
                price = float(ticker_data.get("sell", 0))  # Use sell price for buying
            else:  # sell
                price = float(ticker_data.get("buy", 0))   # Use buy price for selling
            
            if price <= 0:
                logger.error("Invalid price from ticker", pair_id=pair_id, price=price)
                return
            
            # Validate trade with risk manager
            risk_validation = await self.risk_manager.validate_trade(
                user, pair_id, signal_type, amount, price
            )
            
            if not risk_validation["allowed"]:
                logger.warning("Auto-trade rejected by risk manager", 
                             user_id=user_id, pair_id=pair_id, 
                             errors=risk_validation["errors"])
                
                # Notify user about rejected trade
                if self.telegram_bot:
                    await self._notify_user_trade_rejected(
                        telegram_id, pair_id, signal, risk_validation["errors"]
                    )
                return
            
            # Execute the trade
            try:
                if signal_type == "buy":
                    result = await user_api.trade(
                        pair=pair_id,
                        type="buy",
                        price=price,
                        idr_amount=amount
                    )
                    quantity = amount / price
                else:  # sell
                    quantity = amount / price
                    result = await user_api.trade(
                        pair=pair_id,
                        type="sell",
                        price=price,
                        coin_amount=quantity
                    )
                
                if result.get("success") == 1:
                    # Save trade to database
                    await self._save_auto_trade(
                        user_id, pair_id, signal, result["return"]["order_id"], 
                        signal_type, quantity, price, amount
                    )
                    
                    # Notify user about successful trade
                    if self.telegram_bot:
                        await self._notify_user_trade_executed(
                            telegram_id, pair_id, signal, result["return"]["order_id"], 
                            signal_type, quantity, price, amount
                        )
                    
                    logger.info("Auto-trade executed successfully", 
                               user_id=user_id, pair_id=pair_id, 
                               order_id=result["return"]["order_id"],
                               signal_type=signal_type,
                               amount=amount)
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error("Auto-trade execution failed", 
                               user_id=user_id, pair_id=pair_id, error=error_msg)
                    
                    # Notify user about failed trade
                    if self.telegram_bot:
                        await self._notify_user_trade_failed(
                            telegram_id, pair_id, signal, error_msg
                        )
                
            except Exception as e:
                logger.error("Failed to execute auto-trade", 
                           user_id=user_id, pair_id=pair_id, error=str(e))
                
                # Notify user about failed trade
                if self.telegram_bot:
                    await self._notify_user_trade_failed(
                        telegram_id, pair_id, signal, str(e)
                    )
            
        except Exception as e:
            logger.error("Failed to execute auto-trade", 
                       user_id=int(user.id), pair_id=pair_id, error=str(e))  # type: ignore
    
    async def _calculate_auto_trade_amount(self, user: User, pair_id: str, user_settings: Dict[str, Any]) -> float:
        """Calculate trade amount for auto-trading"""
        try:
            # Get user's auto-trading settings
            base_amount = user_settings.get('auto_trade_amount', getattr(user, 'max_trade_amount', 100000.0))
            
            # Apply position sizing based on risk level
            risk_level = getattr(user, 'risk_level', 'medium')
            if risk_level == "low":
                multiplier = 0.5
            elif risk_level == "high":
                multiplier = 1.5
            else:  # medium
                multiplier = 1.0
            
            trade_amount = base_amount * multiplier
            
            # Ensure minimum trade amount (10,000 IDR for Indodax)
            return max(trade_amount, 10000.0)
            
        except Exception as e:
            logger.error("Failed to calculate auto-trade amount", 
                       user_id=int(user.id), pair_id=pair_id, error=str(e))  # type: ignore
            return 0.0
    
    async def _get_daily_auto_trades_count(self, user_id: int) -> int:
        """Get number of auto-trades executed today"""
        try:
            db = get_db()
            try:
                today = datetime.now().date()
                count = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.created_at >= today,
                    Trade.type.in_(["auto_buy", "auto_sell"])
                ).count()
                return count
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to get daily auto-trades count", user_id=user_id, error=str(e))
            return 0
    
    async def _get_user_settings(self, user_id: int) -> Dict[str, Any]:
        """Get user auto-trading settings"""
        try:
            db = get_db()
            try:
                settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                if not settings:
                    return {
                        'auto_trade_amount': 100000.0,
                        'stop_loss_percentage': 5.0,
                        'take_profit_percentage': 10.0
                    }
                
                return {
                    'auto_trade_amount': getattr(settings, 'auto_trade_amount', 100000.0),
                    'stop_loss_percentage': settings.stop_loss_percentage,
                    'take_profit_percentage': settings.take_profit_percentage
                }
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to get user settings", user_id=user_id, error=str(e))
            return {}
    
    async def _get_user_trading_pairs(self, user_id: int) -> List[str]:
        """Get trading pairs for user auto-trading"""
        # For now, return default pairs
        # In the future, this could be user-configurable
        return ["btc_idr", "eth_idr", "usdt_idr"]
    
    async def _has_recent_signal_trade(self, user_id: int, pair_id: str, signal_type: str) -> bool:
        """Check if user has recent trade for similar signal"""
        try:
            db = get_db()
            try:
                # Check for trades in the last 4 hours
                cutoff_time = datetime.now() - timedelta(hours=4)
                
                recent_trade = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.pair_id == pair_id,
                    Trade.type == f"auto_{signal_type}",
                    Trade.created_at >= cutoff_time
                ).first()
                
                return recent_trade is not None
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to check recent signal trade", 
                       user_id=user_id, pair_id=pair_id, error=str(e))
            return False
    
    async def _save_auto_trade(self, user_id: int, pair_id: str, signal: AISignal, 
                              order_id: str, trade_type: str, quantity: float, 
                              price: float, total: float):
        """Save auto-trade to database"""
        try:
            db = get_db()
            try:
                trade = Trade(
                    user_id=user_id,
                    pair_id=pair_id,
                    order_id=order_id,
                    type=f"auto_{trade_type}",  # auto_buy or auto_sell
                    amount=quantity,
                    price=price,
                    total=total,
                    status="pending",
                    created_at=datetime.now()
                )
                
                # Add signal reference
                trade.signal_id = signal.id
                trade.signal_confidence = signal.confidence
                
                db.add(trade)
                db.commit()
                
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to save auto-trade", user_id=user_id, error=str(e))
    
    async def _notify_user_trade_executed(self, telegram_id: int, pair_id: str, 
                                        signal: AISignal, order_id: str, trade_type: str,
                                        quantity: float, price: float, total: float):
        """Notify user about executed auto-trade"""
        try:
            if not self.telegram_bot:
                return
            
            message = f"""
🤖 <b>Auto-Trade Executed!</b>

📊 Signal: {signal.signal_type.upper()}
🪙 Pair: {pair_id.upper()}
📈 Konfidence: {signal.confidence:.1%}
🔄 Order ID: {order_id}

💰 Amount: {quantity:.8f}
💵 Price: {price:,.0f} IDR
💸 Total: {total:,.0f} IDR

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.telegram_bot.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error("Failed to notify user about trade execution", 
                       telegram_id=telegram_id, error=str(e))
    
    async def _notify_user_trade_rejected(self, telegram_id: int, pair_id: str, 
                                        signal: AISignal, errors: List[str]):
        """Notify user about rejected auto-trade"""
        try:
            if not self.telegram_bot:
                return
            
            message = f"""
⚠️ <b>Auto-Trade Rejected</b>

📊 Signal: {signal.signal_type.upper()}
🪙 Pair: {pair_id.upper()}
📈 Confidence: {signal.confidence:.1%}

❌ Reasons:
{chr(10).join(f"• {error}" for error in errors)}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.telegram_bot.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error("Failed to notify user about trade rejection", 
                       telegram_id=telegram_id, error=str(e))
    
    async def _notify_user_trade_failed(self, telegram_id: int, pair_id: str, 
                                      signal: AISignal, error_msg: str):
        """Notify user about failed auto-trade"""
        try:
            if not self.telegram_bot:
                return
            
            message = f"""
❌ <b>Auto-Trade Failed</b>

📊 Signal: {signal.signal_type.upper()}
🪙 Pair: {pair_id.upper()}
📈 Confidence: {signal.confidence:.1%}

🚫 Error: {error_msg}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.telegram_bot.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error("Failed to notify user about trade failure", 
                       telegram_id=telegram_id, error=str(e))
    
    async def enable_auto_trading(self, user_id: int) -> bool:
        """Enable auto-trading for a user"""
        try:
            db = get_db()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    # Use setattr for proper SQLAlchemy attribute assignment
                    setattr(user, 'auto_trading_enabled', True)
                    db.commit()
                    logger.info("Auto-trading enabled", user_id=user_id)
                    return True
                return False
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to enable auto-trading", user_id=user_id, error=str(e))
            return False
    
    async def disable_auto_trading(self, user_id: int) -> bool:
        """Disable auto-trading for a user"""
        try:
            db = get_db()
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if user:
                    # Use setattr for proper SQLAlchemy attribute assignment
                    setattr(user, 'auto_trading_enabled', False)
                    db.commit()
                    logger.info("Auto-trading disabled", user_id=user_id)
                    return True
                return False
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to disable auto-trading", user_id=user_id, error=str(e))
            return False
