"""
Database models and initialization for the trading bot
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from config.settings import settings
import asyncio
import structlog

logger = structlog.get_logger(__name__)

Base = declarative_base()

class User(Base):
    """User model for storing Telegram user information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    is_premium = Column(Boolean, default=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    language = Column(String, default="id")  # id for Indonesian, en for English
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indodax API credentials (encrypted)
    indodax_api_key = Column(Text, nullable=True)
    indodax_secret_key = Column(Text, nullable=True)
    
    # Trading preferences
    auto_trading_enabled = Column(Boolean, default=False)
    risk_level = Column(String, default="medium")  # low, medium, high
    max_trade_amount = Column(Float, default=100000.0)
    auto_trade_amount = Column(Float, default=100000.0)  # Add missing column

class TradingPair(Base):
    """Trading pairs available on Indodax"""
    __tablename__ = "trading_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    pair_id = Column(String, unique=True, index=True)  # e.g., btc_idr
    symbol = Column(String)  # e.g., BTCIDR
    base_currency = Column(String)  # e.g., idr
    traded_currency = Column(String)  # e.g., btc
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())

class Trade(Base):
    """Trading history"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    pair_id = Column(String, index=True)
    order_id = Column(String, unique=True)
    type = Column(String)  # buy, sell
    amount = Column(Float)
    price = Column(Float)
    total = Column(Float)
    status = Column(String)  # pending, completed, cancelled
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)

class AISignal(Base):
    """AI generated trading signals"""
    __tablename__ = "ai_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    pair_id = Column(String, index=True)
    signal_type = Column(String)  # buy, sell, hold
    confidence = Column(Float)  # 0.0 to 1.0
    price_prediction = Column(Float)
    indicators = Column(JSON)  # Technical indicators used
    sentiment_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)

class Portfolio(Base):
    """User portfolio tracking"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    currency = Column(String)  # btc, eth, idr, etc.
    balance = Column(Float, default=0.0)
    locked_balance = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class PriceHistory(Base):
    """Historical price data for analysis"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, index=True)
    pair_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)

class UserSettings(Base):
    """User-specific settings"""
    __tablename__ = "user_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, index=True)
    notifications_enabled = Column(Boolean, default=True)
    stop_loss_percentage = Column(Float, default=5.0)
    take_profit_percentage = Column(Float, default=10.0)
    dca_enabled = Column(Boolean, default=False)
    dca_amount = Column(Float, default=100000.0)
    dca_interval = Column(String, default="daily")  # daily, weekly, monthly
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Database engine and session
engine = None
SessionLocal = None

async def init_database():
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    try:
        logger.info("Connecting to database", url=settings.database_url)
        
        # Create engine
        engine = create_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Create session factory
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

def get_db() -> Session:
    """Get database session"""
    if not SessionLocal:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise

async def close_database():
    """Close database connections"""
    global engine
    if engine:
        engine.dispose()
        logger.info("Database connections closed")
