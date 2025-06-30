"""
Scheduler for automated trading tasks and background jobs
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import asyncio
import structlog

from core.indodax_api import indodax_api
from core.database import get_db, PriceHistory, TradingPair, User, Trade, UserSettings
from ai.signal_generator import SignalGenerator
from ai.sentiment_analyzer import SentimentAnalyzer

logger = structlog.get_logger(__name__)

class TradingScheduler:
    """Scheduler for automated trading tasks"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.signal_generator = SignalGenerator()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def start(self):
        """Start the scheduler"""
        try:
            # Add scheduled jobs
            self._add_jobs()
            
            # Start the scheduler
            self.scheduler.start()
            logger.info("Trading scheduler started successfully")
            
        except Exception as e:
            logger.error("Failed to start scheduler", error=str(e))
            raise
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.scheduler.shutdown()
        logger.info("Trading scheduler shutdown")
    
    def _add_jobs(self):
        """Add all scheduled jobs"""
        
        # Update price data every 5 minutes
        self.scheduler.add_job(
            self.update_price_data,
            IntervalTrigger(minutes=5),
            id="update_price_data",
            max_instances=1,
            coalesce=True
        )
        
        # Generate AI signals every 15 minutes
        self.scheduler.add_job(
            self.generate_ai_signals,
            IntervalTrigger(minutes=15),
            id="generate_ai_signals", 
            max_instances=1,
            coalesce=True
        )
        
        # Update sentiment analysis every 30 minutes
        self.scheduler.add_job(
            self.update_sentiment_analysis,
            IntervalTrigger(minutes=30),
            id="update_sentiment_analysis",
            max_instances=1,
            coalesce=True
        )
        
        # Clean up old data daily at 2 AM
        self.scheduler.add_job(
            self.cleanup_old_data,
            CronTrigger(hour=2, minute=0),
            id="cleanup_old_data",
            max_instances=1
        )
        
        # Generate daily reports at 8 AM
        self.scheduler.add_job(
            self.generate_daily_reports,
            CronTrigger(hour=8, minute=0),
            id="generate_daily_reports",
            max_instances=1
        )
        
        # Check for DCA (Dollar Cost Averaging) trades
        self.scheduler.add_job(
            self.process_dca_trades,
            CronTrigger(hour=9, minute=0),  # Daily at 9 AM
            id="process_dca_trades",
            max_instances=1
        )
        
        # Process auto-trades based on AI signals every 30 minutes
        self.scheduler.add_job(
            self.process_auto_trades,
            IntervalTrigger(minutes=30),
            id="process_auto_trades",
            max_instances=1,
            coalesce=True
        )
        
        # Monitor stop loss/take profit every 5 minutes
        self.scheduler.add_job(
            self.monitor_stop_loss_take_profit,
            IntervalTrigger(minutes=5),
            id="monitor_stop_loss_take_profit",
            max_instances=1,
            coalesce=True
        )
        
        logger.info("Scheduled jobs added successfully")
    
    async def update_price_data(self):
        """Update price data for all active trading pairs"""
        try:
            logger.info("Starting price data update")
            
            # Get all active trading pairs
            db = get_db()
            try:
                pairs = db.query(TradingPair).filter(TradingPair.is_active == True).all()
                
                for pair in pairs:
                    try:
                        # Get current ticker data
                        ticker_data = await indodax_api.get_ticker(str(pair.pair_id))
                        
                        if "ticker" in ticker_data:
                            ticker = ticker_data["ticker"]
                            
                            # Create price history entry
                            price_entry = PriceHistory(
                                pair_id=pair.pair_id,
                                timestamp=datetime.fromtimestamp(ticker["server_time"]),
                                open_price=float(ticker.get("last", 0)),
                                high_price=float(ticker.get("high", 0)),
                                low_price=float(ticker.get("low", 0)),
                                close_price=float(ticker.get("last", 0)),
                                volume=float(ticker.get(f"vol_{pair.traded_currency}", 0))
                            )
                            
                            db.add(price_entry)
                            
                    except Exception as e:
                        logger.error("Failed to update price for pair", pair_id=pair.pair_id, error=str(e))
                        continue
                
                db.commit()
                logger.info("Price data update completed", pairs_updated=len(pairs))
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to update price data", error=str(e))
    
    async def generate_ai_signals(self):
        """Generate AI trading signals for all active pairs"""
        try:
            logger.info("Starting AI signal generation")
            
            db = get_db()
            try:
                pairs = db.query(TradingPair).filter(TradingPair.is_active == True).all()
                
                for pair in pairs:
                    try:
                        # Generate signal for this pair
                        signal = await self.signal_generator.generate_signal(str(pair.pair_id))
                        
                        if signal:
                            logger.info("AI signal generated", 
                                       pair_id=pair.pair_id, 
                                       signal_type=signal.signal_type,
                                       confidence=signal.confidence)
                    
                    except Exception as e:
                        logger.error("Failed to generate signal for pair", 
                                   pair_id=pair.pair_id, error=str(e))
                        continue
                
                logger.info("AI signal generation completed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to generate AI signals", error=str(e))
    
    async def update_sentiment_analysis(self):
        """Update market sentiment analysis"""
        try:
            logger.info("Starting sentiment analysis update")
            
            # Major crypto currencies to analyze
            major_pairs = ["btc_idr", "eth_idr", "bnb_idr", "ada_idr"]
            
            for pair_id in major_pairs:
                try:
                    sentiment = await self.sentiment_analyzer.analyze_sentiment(pair_id)
                    logger.info("Sentiment analysis completed", 
                               pair_id=pair_id, 
                               sentiment_score=sentiment.get("score", 0))
                
                except Exception as e:
                    logger.error("Failed to analyze sentiment for pair", 
                               pair_id=pair_id, error=str(e))
                    continue
            
            logger.info("Sentiment analysis update completed")
            
        except Exception as e:
            logger.error("Failed to update sentiment analysis", error=str(e))
    
    async def cleanup_old_data(self):
        """Clean up old data to maintain database performance"""
        try:
            logger.info("Starting data cleanup")
            
            db = get_db()
            try:
                # Delete price history older than 3 months
                cutoff_date = datetime.now() - timedelta(days=90)
                
                deleted_count = db.query(PriceHistory).filter(
                    PriceHistory.timestamp < cutoff_date
                ).delete()
                
                db.commit()
                
                logger.info("Data cleanup completed", records_deleted=deleted_count)
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to cleanup old data", error=str(e))
    
    async def generate_daily_reports(self):
        """Generate and send daily trading reports"""
        try:
            logger.info("Starting daily report generation")
            
            # This will be implemented with Telegram notifications
            # For now, just log the completion
            
            logger.info("Daily report generation completed")
            
        except Exception as e:
            logger.error("Failed to generate daily reports", error=str(e))
    
    async def process_dca_trades(self):
        """Process Dollar Cost Averaging trades for users"""
        try:
            logger.info("Starting DCA trade processing")
            
            # Import DCA manager to avoid circular imports
            from core.dca_manager import DCAManager
            
            dca_manager = DCAManager()
            await dca_manager.process_dca_trades()
            
            logger.info("DCA trade processing completed")
            
        except Exception as e:
            logger.error("Failed to process DCA trades", error=str(e))
    
    async def process_auto_trades(self):
        """Process auto-trades based on AI signals"""
        try:
            logger.info("Starting auto-trade processing")
            
            # Import auto trader to avoid circular imports
            from core.auto_trader import AutoTrader
            
            auto_trader = AutoTrader()
            await auto_trader.process_auto_trades()
            
            logger.info("Auto-trade processing completed")
            
        except Exception as e:
            logger.error("Failed to process auto-trades", error=str(e))
    
    async def update_price_history(self):
        """Update price history data for all trading pairs"""
        try:
            logger.info("Starting price history update")
            
            db = get_db()
            try:
                # Get all active trading pairs
                pairs = db.query(TradingPair).filter(TradingPair.is_active == True).all()
                
                for pair in pairs:
                    try:
                        # Get current ticker data
                        ticker_pair = str(pair.pair_id).replace("_", "")
                        ticker_data = await indodax_api.get_ticker(ticker_pair)
                        
                        if ticker_data and "ticker" in ticker_data:
                            ticker_info = ticker_data["ticker"]
                            
                            # Create price history record
                            price_record = PriceHistory(
                                pair_id=pair.pair_id,
                                timestamp=datetime.now(),
                                open_price=float(ticker_info.get("last", 0)),
                                high_price=float(ticker_info.get("high", 0)),
                                low_price=float(ticker_info.get("low", 0)),
                                close_price=float(ticker_info.get("last", 0)),
                                volume=float(ticker_info.get("vol_idr", 0))
                            )
                            
                            db.add(price_record)
                            
                        await asyncio.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.error("Failed to update price for pair", 
                                   pair_id=pair.pair_id, error=str(e))
                        continue
                
                db.commit()
                logger.info("Price history update completed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to update price history", error=str(e))
    
    async def monitor_stop_loss_take_profit(self):
        """Monitor and execute stop loss/take profit orders"""
        try:
            logger.info("Starting stop loss/take profit monitoring")
            
            db = get_db()
            try:
                # Get all pending trades with stop loss/take profit
                pending_trades = db.query(Trade).filter(
                    Trade.status == "pending"
                ).all()
                
                for trade in pending_trades:
                    try:
                        # Get user settings
                        user = db.query(User).filter(User.id == trade.user_id).first()
                        if not user or not str(user.indodax_api_key):
                            continue
                        
                        settings = db.query(UserSettings).filter(
                            UserSettings.user_id == user.id
                        ).first()
                        
                        if not settings:
                            continue
                        
                        # Get current price
                        ticker_pair = str(trade.pair_id).replace("_", "")
                        ticker_data = await indodax_api.get_ticker(ticker_pair)
                        
                        if not ticker_data or "ticker" not in ticker_data:
                            continue
                        
                        current_price = float(ticker_data["ticker"]["last"])
                        entry_price = float(str(trade.price))
                        
                        # Check stop loss
                        if str(trade.type) == "buy":
                            stop_loss_price = entry_price * (1 - float(str(settings.stop_loss_percentage)) / 100)
                            take_profit_price = entry_price * (1 + float(str(settings.take_profit_percentage)) / 100)
                            
                            if current_price <= stop_loss_price:
                                await self._execute_stop_loss(user, trade, current_price)
                            elif current_price >= take_profit_price:
                                await self._execute_take_profit(user, trade, current_price)
                        
                        await asyncio.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        logger.error("Failed to monitor trade", 
                                   trade_id=trade.id, error=str(e))
                        continue
                
                logger.info("Stop loss/take profit monitoring completed")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to monitor stop loss/take profit", error=str(e))
    
    async def _execute_stop_loss(self, user: User, trade: Trade, current_price: float):
        """Execute stop loss order"""
        try:
            # Import here to avoid circular imports
            from bot.utils import decrypt_api_key
            from core.indodax_api import IndodaxAPI
            
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Execute sell order
            result = await user_api.trade(
                pair=str(trade.pair_id),
                type="sell",
                price=current_price,
                coin_amount=float(str(trade.amount))
            )
            
            if result.get("success") == 1:
                # Update trade status
                db = get_db()
                try:
                    # Update using SQLAlchemy update method
                    db.query(Trade).filter(Trade.id == trade.id).update({
                        Trade.status: "stop_loss_executed",
                        Trade.completed_at: datetime.now()
                    })
                    db.commit()
                    
                    logger.info("Stop loss executed", 
                               user_id=user.id, trade_id=trade.id, price=current_price)
                    
                    # Note: Telegram notification would be handled by main bot instance
                    
                finally:
                    db.close()
            
        except Exception as e:
            logger.error("Failed to execute stop loss", 
                       user_id=user.id, trade_id=trade.id, error=str(e))
    
    async def _execute_take_profit(self, user: User, trade: Trade, current_price: float):
        """Execute take profit order"""
        try:
            # Import here to avoid circular imports
            from bot.utils import decrypt_api_key
            from core.indodax_api import IndodaxAPI
            
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            user_api = IndodaxAPI(api_key, secret_key)
            
            # Execute sell order
            result = await user_api.trade(
                pair=str(trade.pair_id),
                type="sell",
                price=current_price,
                coin_amount=float(str(trade.amount))
            )
            
            if result.get("success") == 1:
                # Update trade status
                db = get_db()
                try:
                    # Update using SQLAlchemy update method
                    db.query(Trade).filter(Trade.id == trade.id).update({
                        Trade.status: "take_profit_executed",
                        Trade.completed_at: datetime.now()
                    })
                    db.commit()
                    
                    logger.info("Take profit executed", 
                               user_id=user.id, trade_id=trade.id, price=current_price)
                    
                    # Note: Telegram notification would be handled by main bot instance
                    
                finally:
                    db.close()
            
        except Exception as e:
            logger.error("Failed to execute take profit", 
                       user_id=user.id, trade_id=trade.id, error=str(e))
    
    async def _notify_stop_loss_executed(self, user: User, trade: Trade, price: float):
        """Notify user about stop loss execution - handled by main bot instance"""
        try:
            # This functionality should be handled by the main bot instance
            # when it receives notification about executed stop loss
            logger.info("Stop loss notification needed", 
                       user_id=user.id, trade_id=trade.id, price=price)
            
        except Exception as e:
            logger.error("Failed to prepare stop loss notification", user_id=user.id, error=str(e))
    
    async def _notify_take_profit_executed(self, user: User, trade: Trade, price: float):
        """Notify user about take profit execution - handled by main bot instance"""
        try:
            # This functionality should be handled by the main bot instance
            # when it receives notification about executed take profit
            logger.info("Take profit notification needed", 
                       user_id=user.id, trade_id=trade.id, price=price)
            
        except Exception as e:
            logger.error("Failed to prepare take profit notification", user_id=user.id, error=str(e))

def init_scheduler() -> TradingScheduler:
    """Initialize and return the trading scheduler"""
    return TradingScheduler()
