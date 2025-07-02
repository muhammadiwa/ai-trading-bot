"""
Dollar Cost Averaging (DCA) implementation for systematic investment
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

from core.database import get_db, User, UserSettings, Trade
from core.indodax_api import IndodaxAPI
from core.risk_manager import RiskManager
from bot.utils import decrypt_api_key
from bot.telegram_bot import TelegramBot

logger = structlog.get_logger(__name__)

class DCAManager:
    """Dollar Cost Averaging manager for systematic investments"""
    
    def __init__(self, telegram_bot: Optional[TelegramBot] = None):
        self.risk_manager = RiskManager()
        self.telegram_bot = telegram_bot
    
    async def process_dca_trades(self):
        """Process DCA trades for all eligible users"""
        try:
            logger.info("Starting DCA trade processing")
            
            db = get_db()
            try:
                # Get users with DCA enabled
                dca_users = db.query(User).join(UserSettings).filter(
                    UserSettings.dca_enabled == True,
                    User.is_active == True,
                    User.indodax_api_key.isnot(None)
                ).all()
                
                if not dca_users:
                    logger.info("No users with DCA enabled")
                    return
                
                logger.info("Processing DCA trades for users", count=len(dca_users))
                
                # Process each user
                for user in dca_users:
                    try:
                        await self._process_user_dca(user)
                    except Exception as e:
                        logger.error("Failed to process DCA for user", 
                                   user_id=user.id, error=str(e))
                        continue
                
                logger.info("DCA trade processing completed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to process DCA trades", error=str(e))
    
    async def _process_user_dca(self, user: User):
        """Process DCA for a specific user"""
        try:
            # Cast user.id to int to fix type checking
            user_id = int(user.id)  # type: ignore
            
            # Get user DCA settings
            dca_settings = await self._get_dca_settings(user_id)
            
            if not dca_settings:
                logger.warning("No DCA settings found", user_id=user_id)
                return
            
            # Check if it's time for DCA trade
            if not await self._is_dca_due(user_id, dca_settings):
                logger.debug("DCA not due yet", user_id=user_id)
                return
            
            # Get DCA pairs for user
            dca_pairs = await self._get_dca_pairs(user_id)
            
            total_amount_per_pair = dca_settings['dca_amount'] / len(dca_pairs)
            
            for pair_id in dca_pairs:
                try:
                    await self._execute_dca_trade(user, pair_id, total_amount_per_pair, dca_settings)
                except Exception as e:
                    logger.error("Failed to execute DCA trade for pair", 
                               user_id=user_id, pair_id=pair_id, error=str(e))
                    continue
            
        except Exception as e:
            logger.error("Failed to process user DCA", user_id=user.id, error=str(e))
    
    async def _execute_dca_trade(self, user: User, pair_id: str, amount: float, dca_settings: Dict[str, Any]):
        """Execute DCA trade for a specific pair"""
        try:
            # Cast user attributes to proper types
            user_id = int(user.id)  # type: ignore
            telegram_id = int(user.telegram_id)  # type: ignore
            
            # Get current market price
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Get ticker for current price
            ticker_pair = pair_id.replace("_", "") if "_" in pair_id else f"{pair_id}idr"
            ticker = await user_api.get_ticker(ticker_pair)
            
            if not ticker or "ticker" not in ticker:
                logger.error("Failed to get ticker data for DCA", pair_id=pair_id)
                return
            
            ticker_data = ticker["ticker"]
            price = float(ticker_data.get("sell", 0))  # Always buy for DCA
            
            if price <= 0:
                logger.error("Invalid price from ticker for DCA", pair_id=pair_id, price=price)
                return
            
            # Validate trade with risk manager
            risk_validation = await self.risk_manager.validate_trade(
                user, pair_id, "buy", amount, price
            )
            
            if not risk_validation["allowed"]:
                logger.warning("DCA trade rejected by risk manager", 
                             user_id=user_id, pair_id=pair_id, 
                             errors=risk_validation["errors"])
                
                # Notify user about rejected DCA
                if self.telegram_bot:
                    await self._notify_dca_rejected(
                        telegram_id, pair_id, amount, risk_validation["errors"]
                    )
                return
            
            # Execute DCA buy order
            try:
                result = await user_api.trade(
                    pair=pair_id,
                    type="buy",
                    price=price,
                    idr_amount=amount
                )
                
                if result.get("success") == 1:
                    quantity = amount / price
                    
                    # Save DCA trade to database
                    await self._save_dca_trade(
                        user_id, pair_id, result["return"]["order_id"], 
                        quantity, price, amount
                    )
                    
                    # Update last DCA timestamp
                    await self._update_last_dca_timestamp(user_id)
                    
                    # Notify user about successful DCA
                    if self.telegram_bot:
                        await self._notify_dca_executed(
                            telegram_id, pair_id, result["return"]["order_id"],
                            quantity, price, amount
                        )
                    
                    logger.info("DCA trade executed successfully", 
                               user_id=user_id, pair_id=pair_id, 
                               order_id=result["return"]["order_id"],
                               amount=amount)
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error("DCA trade execution failed", 
                               user_id=user_id, pair_id=pair_id, error=error_msg)
                    
                    # Notify user about failed DCA
                    if self.telegram_bot:
                        await self._notify_dca_failed(
                            telegram_id, pair_id, amount, error_msg
                        )
                
            except Exception as e:
                logger.error("Failed to execute DCA trade", 
                           user_id=user_id, pair_id=pair_id, error=str(e))
                
                # Notify user about failed DCA
                if self.telegram_bot:
                    await self._notify_dca_failed(
                        telegram_id, pair_id, amount, str(e)
                    )
            
        except Exception as e:
            # Use user.id for error logging since user_id might not be defined
            logger.error("Failed to execute DCA trade", 
                       user_id=int(user.id), pair_id=pair_id, error=str(e))  # type: ignore
    
    async def _get_dca_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user DCA settings"""
        try:
            db = get_db()
            try:
                settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                
                if not settings or getattr(settings, 'dca_enabled', False) is not True:
                    return None
                
                return {
                    'dca_amount': getattr(settings, 'dca_amount', 0.0),
                    'dca_interval': getattr(settings, 'dca_interval', 'daily'),
                    'dca_enabled': getattr(settings, 'dca_enabled', False)
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get DCA settings", user_id=user_id, error=str(e))
            return None
    
    async def _is_dca_due(self, user_id: int, dca_settings: Dict[str, Any]) -> bool:
        """Check if DCA is due for a user"""
        try:
            db = get_db()
            try:
                # Get last DCA trade
                last_dca = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.type == "dca_buy"
                ).order_by(Trade.created_at.desc()).first()
                
                if not last_dca:
                    # First DCA trade
                    return True
                
                # Check interval
                interval = dca_settings['dca_interval']
                now = datetime.now()
                last_created_at = getattr(last_dca, 'created_at', None)
                
                if not last_created_at:
                    return True
                
                if interval == "daily":
                    next_dca = last_created_at + timedelta(days=1)
                elif interval == "weekly":
                    next_dca = last_created_at + timedelta(weeks=1)
                elif interval == "monthly":
                    next_dca = last_created_at + timedelta(days=30)
                else:
                    logger.warning("Unknown DCA interval", interval=interval)
                    return False
                
                return now >= next_dca
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to check if DCA is due", user_id=user_id, error=str(e))
            return False
    
    async def _get_dca_pairs(self, user_id: int) -> List[str]:
        """Get trading pairs for DCA"""
        # For now, return default DCA pairs
        # In the future, this could be user-configurable
        return ["btc_idr", "eth_idr"]
    
    async def _save_dca_trade(self, user_id: int, pair_id: str, order_id: str, 
                             quantity: float, price: float, total: float):
        """Save DCA trade to database"""
        try:
            db = get_db()
            try:
                trade = Trade(
                    user_id=user_id,
                    pair_id=pair_id,
                    order_id=order_id,
                    type="dca_buy",
                    amount=quantity,
                    price=price,
                    total=total,
                    status="pending",
                    created_at=datetime.now()
                )
                
                db.add(trade)
                db.commit()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to save DCA trade", user_id=user_id, error=str(e))
    
    async def _update_last_dca_timestamp(self, user_id: int):
        """Update the last DCA timestamp for a user"""
        try:
            db = get_db()
            try:
                settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                if settings:
                    # Use setattr to properly set SQLAlchemy attributes
                    setattr(settings, 'updated_at', datetime.now())
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error("Failed to update DCA timestamp", user_id=user_id, error=str(e))
    
    async def _notify_dca_executed(self, telegram_id: int, pair_id: str, 
                                  order_id: str, quantity: float, price: float, total: float):
        """Notify user about executed DCA trade"""
        try:
            if not self.telegram_bot:
                return
            
            message = f"""
💰 <b>DCA Trade Executed!</b>

🪙 Pair: {pair_id.upper()}
🔄 Order ID: {order_id}

💰 Amount: {quantity:.8f}
💵 Price: {price:,.0f} IDR
💸 Total: {total:,.0f} IDR

📅 Next DCA: Check your schedule
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💡 DCA membantu mengurangi volatility dengan investasi berkala!
"""
            
            await self.telegram_bot.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error("Failed to notify user about DCA execution", 
                       telegram_id=telegram_id, error=str(e))
    
    async def _notify_dca_rejected(self, telegram_id: int, pair_id: str, 
                                  amount: float, errors: List[str]):
        """Notify user about rejected DCA trade"""
        try:
            if not self.telegram_bot:
                return
            
            message = f"""
⚠️ <b>DCA Trade Rejected</b>

🪙 Pair: {pair_id.upper()}
💸 Amount: {amount:,.0f} IDR

❌ Reasons:
{chr(10).join(f"• {error}" for error in errors)}

💡 Periksa saldo dan pengaturan DCA Anda.
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.telegram_bot.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error("Failed to notify user about DCA rejection", 
                       telegram_id=telegram_id, error=str(e))
    
    async def _notify_dca_failed(self, telegram_id: int, pair_id: str, 
                                amount: float, error_msg: str):
        """Notify user about failed DCA trade"""
        try:
            if not self.telegram_bot:
                return
            
            message = f"""
❌ <b>DCA Trade Failed</b>

🪙 Pair: {pair_id.upper()}
💸 Amount: {amount:,.0f} IDR

🚫 Error: {error_msg}

💡 DCA akan dicoba lagi pada jadwal berikutnya.
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.telegram_bot.bot.send_message(
                chat_id=telegram_id,
                text=message,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error("Failed to notify user about DCA failure", 
                       telegram_id=telegram_id, error=str(e))
    
    async def enable_dca(self, user_id: int, amount: float, interval: str) -> bool:
        """Enable DCA for a user"""
        try:
            if interval not in ["daily", "weekly", "monthly"]:
                return False
            
            if amount < 10000:  # Minimum 10k IDR
                return False
            
            db = get_db()
            try:
                settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                
                if not settings:
                    settings = UserSettings(user_id=user_id)
                    db.add(settings)
                
                # Use setattr for proper SQLAlchemy attribute assignment
                setattr(settings, 'dca_enabled', True)
                setattr(settings, 'dca_amount', amount)
                setattr(settings, 'dca_interval', interval)
                setattr(settings, 'updated_at', datetime.now())
                
                db.commit()
                
                logger.info("DCA enabled", user_id=user_id, amount=amount, interval=interval)
                return True
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to enable DCA", user_id=user_id, error=str(e))
            return False
    
    async def disable_dca(self, user_id: int) -> bool:
        """Disable DCA for a user"""
        try:
            db = get_db()
            try:
                settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
                
                if settings:
                    # Use setattr for proper SQLAlchemy attribute assignment
                    setattr(settings, 'dca_enabled', False)
                    setattr(settings, 'updated_at', datetime.now())
                    db.commit()
                    
                    logger.info("DCA disabled", user_id=user_id)
                    return True
                
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to disable DCA", user_id=user_id, error=str(e))
            return False
    
    async def get_dca_status(self, user_id: int) -> Dict[str, Any]:
        """Get DCA status for a user"""
        try:
            dca_settings = await self._get_dca_settings(user_id)
            
            if not dca_settings:
                return {
                    "enabled": False,
                    "amount": 0.0,
                    "interval": "daily",
                    "next_dca": None,
                    "total_invested": 0.0,
                    "trade_count": 0
                }
            
            # Get DCA stats
            db = get_db()
            try:
                dca_trades = db.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.type == "dca_buy"
                ).all()
                
                total_invested = sum(trade.total for trade in dca_trades)
                trade_count = len(dca_trades)
                
                # Calculate next DCA date
                last_dca = max(dca_trades, key=lambda t: getattr(t, 'created_at')) if dca_trades else None
                next_dca = None
                
                if last_dca:
                    interval = dca_settings['dca_interval']
                    last_created_at = getattr(last_dca, 'created_at', None)
                    
                    if last_created_at:
                        if interval == "daily":
                            next_dca = last_created_at + timedelta(days=1)
                        elif interval == "weekly":
                            next_dca = last_created_at + timedelta(weeks=1)
                        elif interval == "monthly":
                            next_dca = last_created_at + timedelta(days=30)
                
                return {
                    "enabled": True,
                    "amount": dca_settings['dca_amount'],
                    "interval": dca_settings['dca_interval'],
                    "next_dca": next_dca.isoformat() if next_dca else None,
                    "total_invested": total_invested,
                    "trade_count": trade_count
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get DCA status", user_id=user_id, error=str(e))
            return {
                "enabled": False,
                "amount": 0.0,
                "interval": "daily",
                "next_dca": None,
                "total_invested": 0.0,
                "trade_count": 0
            }
