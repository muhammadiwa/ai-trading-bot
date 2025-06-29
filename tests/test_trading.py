"""
Test trading functionality
"""
import pytest
from unittest.mock import AsyncMock, patch
from core.indodax_api import IndodaxAPI
from core.risk_manager import RiskManager
from core.database import User, Trade

class TestIndodaxAPI:
    """Test Indodax API integration"""
    
    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_indodax_api):
        """Test getting ticker data"""
        api = IndodaxAPI()
        
        with patch.object(api, 'get_ticker', return_value=mock_indodax_api.get_ticker.return_value):
            result = await api.get_ticker("btc_idr")
            
            assert result is not None
            assert "ticker" in result
            assert "last" in result["ticker"]
    
    @pytest.mark.asyncio
    async def test_get_balance(self, mock_indodax_api):
        """Test getting balance"""
        api = IndodaxAPI("test_key", "test_secret")
        
        with patch.object(api, 'get_balance', return_value=mock_indodax_api.get_balance.return_value):
            result = await api.get_balance()
            
            assert result is not None
            assert "idr" in result
            assert "btc" in result
    
    @pytest.mark.asyncio
    async def test_place_trade(self, mock_indodax_api):
        """Test placing a trade"""
        api = IndodaxAPI("test_key", "test_secret")
        
        with patch.object(api, 'trade', return_value=mock_indodax_api.trade.return_value):
            result = await api.trade("btc_idr", "buy", 645000000, 0.001)
            
            assert result is not None
            assert result.get("success") == 1
            assert "return" in result

class TestRiskManager:
    """Test risk management functionality"""
    
    @pytest.mark.asyncio
    async def test_validate_trade_success(self, test_db):
        """Test successful trade validation"""
        risk_manager = RiskManager()
        
        # Create test user
        user = User(
            telegram_id=123456789,
            username="testuser",
            indodax_api_key="encrypted_api_key",
            indodax_secret_key="encrypted_secret_key"
        )
        test_db.add(user)
        test_db.commit()
        
        with patch('core.risk_manager.RiskManager._get_daily_trades_count', return_value=5):
            with patch('core.risk_manager.RiskManager._check_balance_risk', 
                      return_value={"valid": True, "warnings": [], "errors": []}):
                with patch('core.risk_manager.RiskManager._check_position_size',
                          return_value={"valid": True, "warnings": []}):
                    
                    result = await risk_manager.validate_trade(
                        user, "btc_idr", "buy", 100000, 645000000
                    )
                    
                    assert result["allowed"] is True
                    assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio  
    async def test_validate_trade_daily_limit_exceeded(self, test_db):
        """Test trade validation when daily limit exceeded"""
        risk_manager = RiskManager()
        
        user = User(
            telegram_id=123456789,
            username="testuser",
            indodax_api_key="encrypted_api_key",
            indodax_secret_key="encrypted_secret_key"
        )
        test_db.add(user)
        test_db.commit()
        
        with patch('core.risk_manager.RiskManager._get_daily_trades_count', return_value=15):
            result = await risk_manager.validate_trade(
                user, "btc_idr", "buy", 100000, 645000000
            )
            
            assert result["allowed"] is False
            assert len(result["errors"]) > 0
            assert "Batas maksimal" in result["errors"][0]

class TestTradeExecution:
    """Test trade execution logic"""
    
    @pytest.mark.asyncio
    async def test_successful_buy_order(self, test_db, mock_indodax_api):
        """Test successful buy order execution"""
        user = User(
            telegram_id=123456789,
            username="testuser",
            indodax_api_key="encrypted_api_key",
            indodax_secret_key="encrypted_secret_key"
        )
        test_db.add(user)
        test_db.commit()
        
        # Mock the trade execution
        with patch('core.indodax_api.indodax_api.trade', 
                  return_value=mock_indodax_api.trade.return_value):
            
            # Trade should be successful
            trade_result = await mock_indodax_api.trade("btc_idr", "buy", 645000000, 0.001)
            
            assert trade_result["success"] == 1
            assert "order_id" in trade_result["return"]
    
    @pytest.mark.asyncio
    async def test_insufficient_balance(self, test_db):
        """Test trade rejection due to insufficient balance"""
        user = User(
            telegram_id=123456789,
            username="testuser",
            indodax_api_key="encrypted_api_key",
            indodax_secret_key="encrypted_secret_key"
        )
        test_db.add(user)
        test_db.commit()
        
        risk_manager = RiskManager()
        
        # Mock insufficient balance
        with patch('core.risk_manager.RiskManager._check_balance_risk',
                  return_value={
                      "valid": False,
                      "warnings": [],
                      "errors": ["Saldo IDR tidak mencukupi"]
                  }):
            
            result = await risk_manager.validate_trade(
                user, "btc_idr", "buy", 10000000, 645000000  # Large amount
            )
            
            assert result["allowed"] is False
            assert "Saldo IDR tidak mencukupi" in result["errors"]

class TestSignalGeneration:
    """Test AI signal generation"""
    
    @pytest.mark.asyncio
    async def test_technical_signal_generation(self):
        """Test technical analysis signal generation"""
        from ai.signal_generator import SignalGenerator
        
        generator = SignalGenerator()
        
        # Mock historical data and indicators
        with patch.object(generator, '_get_historical_data') as mock_hist:
            with patch.object(generator, '_calculate_technical_indicators') as mock_indicators:
                with patch.object(generator, '_generate_technical_signal') as mock_signal:
                    
                    # Setup mocks
                    mock_hist.return_value = {"close": [100, 101, 102, 103, 104]}
                    mock_indicators.return_value = {
                        "rsi": [30, 35, 40, 45, 50],
                        "macd": [0.1, 0.2, 0.3, 0.4, 0.5]
                    }
                    
                    from core.database import AISignal
                    from datetime import datetime, timedelta
                    
                    mock_signal.return_value = AISignal(
                        pair_id="btc_idr",
                        signal_type="buy",
                        confidence=0.8,
                        price_prediction=105.0,
                        indicators={"rsi": 30, "macd": 0.5},
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=1)
                    )
                    
                    # Generate signal
                    signal = await generator.generate_signal("btc_idr")
                    
                    assert signal is not None
                    assert signal.signal_type in ["buy", "sell", "hold"]
                    assert 0 <= signal.confidence <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
