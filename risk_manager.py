from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RiskManager:
    """
    Kelas untuk manajemen risiko trading
    """
    
    def __init__(self, config: Dict):
        self.stop_loss_percent = config.get('stop_loss_percent', 5.0)
        self.take_profit_percent = config.get('take_profit_percent', 10.0)
        self.max_daily_trades = config.get('max_daily_trades', 10)
        self.max_position_size = config.get('max_position_size', 0.1)  # 10% of portfolio
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 5% of portfolio
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% risk per trade
        
        # Track trading statistics
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
        
    def reset_daily_counters(self):
        """Reset counter harian"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
    
    def calculate_position_size(self, account_balance: float, current_price: float, 
                              stop_loss_price: float) -> float:
        """
        Menghitung ukuran posisi berdasarkan manajemen risiko
        """
        try:
            # Reset daily counters jika perlu
            self.reset_daily_counters()
            
            # Risk amount (berapa banyak yang bersedia hilang)
            risk_amount = account_balance * self.risk_per_trade
            
            # Price difference untuk stop loss
            price_diff = abs(current_price - stop_loss_price)
            
            if price_diff == 0:
                return 0
            
            # Position size berdasarkan risk
            position_size = risk_amount / price_diff
            
            # Batasi berdasarkan max position size
            max_position_value = account_balance * self.max_position_size
            max_position_size = max_position_value / current_price
            
            # Ambil yang terkecil
            final_position_size = min(position_size, max_position_size)
            
            return max(0, final_position_size)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0
    
    def calculate_stop_loss(self, entry_price: float, order_type: str) -> float:
        """
        Menghitung harga stop loss
        """
        if order_type.lower() == 'buy':
            # Stop loss di bawah entry price untuk buy order
            return entry_price * (1 - self.stop_loss_percent / 100)
        else:
            # Stop loss di atas entry price untuk sell order
            return entry_price * (1 + self.stop_loss_percent / 100)
    
    def calculate_take_profit(self, entry_price: float, order_type: str) -> float:
        """
        Menghitung harga take profit
        """
        if order_type.lower() == 'buy':
            # Take profit di atas entry price untuk buy order
            return entry_price * (1 + self.take_profit_percent / 100)
        else:
            # Take profit di bawah entry price untuk sell order
            return entry_price * (1 - self.take_profit_percent / 100)
    
    def should_execute_trade(self, trade_data: Dict, account_balance: float) -> Dict[str, any]:
        """
        Menentukan apakah trade harus dieksekusi berdasarkan aturan risk management
        """
        self.reset_daily_counters()
        
        result = {
            'should_trade': True,
            'reason': '',
            'recommended_position_size': 0,
            'stop_loss_price': 0,
            'take_profit_price': 0
        }
        
        # Check daily trade limit
        if self.daily_trades_count >= self.max_daily_trades:
            result['should_trade'] = False
            result['reason'] = f"Daily trade limit reached ({self.max_daily_trades})"
            return result
        
        # Check daily loss limit
        daily_loss_percent = abs(self.daily_pnl) / account_balance if account_balance > 0 else 0
        if self.daily_pnl < 0 and daily_loss_percent >= self.max_daily_loss:
            result['should_trade'] = False
            result['reason'] = f"Daily loss limit reached ({self.max_daily_loss*100:.1f}%)"
            return result
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            result['should_trade'] = False
            result['reason'] = f"Max consecutive losses reached ({self.max_consecutive_losses})"
            return result
        
        # Calculate position parameters
        entry_price = trade_data.get('price', 0)
        order_type = trade_data.get('order_type', 'buy')
        
        if entry_price <= 0:
            result['should_trade'] = False
            result['reason'] = "Invalid entry price"
            return result
        
        # Calculate stop loss and take profit
        stop_loss_price = self.calculate_stop_loss(entry_price, order_type)
        take_profit_price = self.calculate_take_profit(entry_price, order_type)
        
        # Calculate recommended position size
        position_size = self.calculate_position_size(account_balance, entry_price, stop_loss_price)
        
        if position_size <= 0:
            result['should_trade'] = False
            result['reason'] = "Position size too small or invalid"
            return result
        
        result['recommended_position_size'] = position_size
        result['stop_loss_price'] = stop_loss_price
        result['take_profit_price'] = take_profit_price
        
        return result
    
    def update_trade_result(self, trade_result: Dict):
        """
        Update hasil trading untuk tracking
        """
        self.reset_daily_counters()
        
        # Update daily counters
        self.daily_trades_count += 1
        
        pnl = trade_result.get('pnl', 0)
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def get_risk_metrics(self, account_balance: float) -> Dict:
        """
        Mendapatkan metrik risiko saat ini
        """
        self.reset_daily_counters()
        
        daily_loss_percent = abs(self.daily_pnl) / account_balance if account_balance > 0 else 0
        
        return {
            'daily_trades_count': self.daily_trades_count,
            'daily_trades_remaining': max(0, self.max_daily_trades - self.daily_trades_count),
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': (self.daily_pnl / account_balance * 100) if account_balance > 0 else 0,
            'daily_loss_limit_remaining': max(0, self.max_daily_loss - daily_loss_percent),
            'consecutive_losses': self.consecutive_losses,
            'consecutive_losses_remaining': max(0, self.max_consecutive_losses - self.consecutive_losses),
            'can_trade': (
                self.daily_trades_count < self.max_daily_trades and
                daily_loss_percent < self.max_daily_loss and
                self.consecutive_losses < self.max_consecutive_losses
            )
        }
    
    def check_stop_loss_take_profit(self, open_positions: List[Dict], current_prices: Dict) -> List[Dict]:
        """
        Check apakah ada posisi yang harus ditutup karena stop loss atau take profit
        """
        actions = []
        
        for position in open_positions:
            pair = position.get('pair', '')
            entry_price = position.get('entry_price', 0)
            stop_loss = position.get('stop_loss_price', 0)
            take_profit = position.get('take_profit_price', 0)
            order_type = position.get('order_type', '')
            current_price = current_prices.get(pair, 0)
            
            if current_price <= 0 or entry_price <= 0:
                continue
            
            action = None
            reason = ""
            
            if order_type.lower() == 'buy':
                # Untuk buy position
                if current_price <= stop_loss:
                    action = 'sell'
                    reason = f"Stop loss triggered at {current_price} (SL: {stop_loss})"
                elif current_price >= take_profit:
                    action = 'sell'
                    reason = f"Take profit triggered at {current_price} (TP: {take_profit})"
            else:
                # Untuk sell position
                if current_price >= stop_loss:
                    action = 'buy'
                    reason = f"Stop loss triggered at {current_price} (SL: {stop_loss})"
                elif current_price <= take_profit:
                    action = 'buy'
                    reason = f"Take profit triggered at {current_price} (TP: {take_profit})"
            
            if action:
                actions.append({
                    'pair': pair,
                    'action': action,
                    'current_price': current_price,
                    'reason': reason,
                    'position': position
                })
        
        return actions
    
    def calculate_portfolio_risk(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """
        Menghitung risiko portfolio
        """
        try:
            total_value = portfolio_data.get('total_value', 0)
            positions = portfolio_data.get('positions', [])
            
            if total_value <= 0:
                return {'total_risk': 0, 'risk_per_asset': {}}
            
            risk_metrics = {
                'total_risk': 0,
                'risk_per_asset': {},
                'concentration_risk': 0,
                'max_drawdown_risk': 0
            }
            
            # Calculate risk per asset
            for position in positions:
                asset = position.get('asset', '')
                value = position.get('value', 0)
                volatility = market_data.get(f"{asset}_volatility", 0.02)  # Default 2% daily volatility
                
                asset_risk = (value / total_value) * volatility
                risk_metrics['risk_per_asset'][asset] = asset_risk
                risk_metrics['total_risk'] += asset_risk
            
            # Concentration risk (largest position as % of portfolio)
            if positions:
                largest_position = max(positions, key=lambda x: x.get('value', 0))
                risk_metrics['concentration_risk'] = largest_position.get('value', 0) / total_value
            
            return risk_metrics
            
        except Exception as e:
            print(f"Error calculating portfolio risk: {e}")
            return {'total_risk': 0, 'risk_per_asset': {}}
    
    def get_trading_recommendations(self, market_conditions: Dict, portfolio_data: Dict) -> List[str]:
        """
        Memberikan rekomendasi trading berdasarkan kondisi risiko
        """
        recommendations = []
        
        risk_metrics = self.get_risk_metrics(portfolio_data.get('total_value', 0))
        
        # Daily trade recommendations
        if risk_metrics['daily_trades_remaining'] <= 2:
            recommendations.append("⚠️ Mendekati batas trading harian - pertimbangkan untuk lebih selektif")
        
        # Daily loss recommendations
        if risk_metrics['daily_pnl'] < 0:
            recommendations.append("📉 Kerugian harian terdeteksi - pertimbangkan untuk mengurangi ukuran posisi")
        
        # Consecutive losses
        if risk_metrics['consecutive_losses'] >= 2:
            recommendations.append("🔴 Kerugian berturut-turut - pertimbangkan untuk pause trading sementara")
        
        # Market volatility recommendations
        volatility = market_conditions.get('volatility', 0)
        if volatility > 0.05:  # High volatility
            recommendations.append("⚡ Volatilitas tinggi terdeteksi - gunakan stop loss yang lebih ketat")
        
        # Portfolio concentration
        portfolio_risk = self.calculate_portfolio_risk(portfolio_data, market_conditions)
        if portfolio_risk.get('concentration_risk', 0) > 0.5:
            recommendations.append("⚖️ Konsentrasi portfolio tinggi - pertimbangkan diversifikasi")
        
        return recommendations
