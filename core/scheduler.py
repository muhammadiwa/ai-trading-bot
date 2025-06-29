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
from core.database import get_db, PriceHistory, TradingPair
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
                        ticker_data = await indodax_api.get_ticker(pair.pair_id)
                        
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
                        signal = await self.signal_generator.generate_signal(pair.pair_id)
                        
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
            
            # This will be implemented with user settings and auto-trading
            # For now, just log the completion
            
            logger.info("DCA trade processing completed")
            
        except Exception as e:
            logger.error("Failed to process DCA trades", error=str(e))

def init_scheduler() -> TradingScheduler:
    """Initialize and return the trading scheduler"""
    return TradingScheduler()
