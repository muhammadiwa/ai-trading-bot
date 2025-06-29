"""
Test configuration and fixtures
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database import Base, get_db

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_trading_bot.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_db():
    """Create a test database session."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Drop tables after test
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def mock_indodax_api():
    """Mock Indodax API responses."""
    mock_api = AsyncMock()
    
    # Mock get_ticker response
    mock_api.get_ticker.return_value = {
        "ticker": {
            "high": "650000000",
            "low": "630000000", 
            "vol_btc": "100.5",
            "vol_idr": "65000000000",
            "last": "645000000",
            "buy": "644000000",
            "sell": "645000000",
            "server_time": 1640995200
        }
    }
    
    # Mock get_balance response
    mock_api.get_balance.return_value = {
        "idr": "1000000",
        "btc": "0.001",
        "eth": "0.01"
    }
    
    # Mock trade response
    mock_api.trade.return_value = {
        "success": 1,
        "return": {
            "order_id": "12345",
            "balance": {
                "idr": "900000",
                "btc": "0.0011"
            }
        }
    }
    
    return mock_api

@pytest.fixture
def mock_telegram_user():
    """Mock Telegram user object."""
    user = MagicMock()
    user.id = 123456789
    user.username = "testuser"
    user.first_name = "Test"
    user.last_name = "User"
    return user

@pytest.fixture
def mock_telegram_message():
    """Mock Telegram message object."""
    message = MagicMock()
    message.from_user.id = 123456789
    message.from_user.username = "testuser"
    message.text = "/start"
    message.answer = AsyncMock()
    return message
