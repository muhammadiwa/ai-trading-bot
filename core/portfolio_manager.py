"""
Portfolio manager for tracking and analyzing user portfolios
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import structlog

from core.database import get_db, User, Portfolio, Trade, PriceHistory
from core.indodax_api import indodax_api
from bot.utils import decrypt_api_key, format_currency

logger = structlog.get_logger(__name__)

class PortfolioManager:
    """Manage user portfolios and provide analytics"""
    
    def __init__(self):
        pass
    
    async def update_user_portfolio(self, user: User) -> Dict[str, Any]:
        """Update user portfolio data from Indodax"""
        try:
            logger.info("Updating portfolio", user_id=user.id)
            
            # Get balance from Indodax
            api_key = decrypt_api_key(str(user.indodax_api_key))
            secret_key = decrypt_api_key(str(user.indodax_secret_key))
            
            user_api = indodax_api.__class__(api_key, secret_key)
            info_data = await user_api.get_info()
            
            if info_data.get('success') != 1:
                raise Exception("Failed to get account info from Indodax")
            
            balance_data = info_data.get('return', {}).get('balance', {})
            
            # Update portfolio in database
            db = get_db()
            try:
                # Clear existing portfolio data
                db.query(Portfolio).filter(Portfolio.user_id == user.id).delete()
                
                # Add new portfolio data
                for currency, balance in balance_data.items():
                    if float(balance) > 0:
                        portfolio_entry = Portfolio(
                            user_id=user.id,
                            currency=currency,
                            balance=float(balance),
                            locked_balance=0.0  # Would need to get this from locked_balance if available
                        )
                        db.add(portfolio_entry)
                
                db.commit()
                
                logger.info("Portfolio updated successfully", user_id=user.id)
                
                return await self.get_portfolio_summary(user)
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to update portfolio", user_id=user.id, error=str(e))
            raise
    
    async def get_portfolio_summary(self, user: User) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            db = get_db()
            try:
                portfolio_entries = db.query(Portfolio).filter(Portfolio.user_id == user.id).all()
                
                if not portfolio_entries:
                    return {
                        "total_value_idr": 0.0,
                        "assets": [],
                        "allocation": {},
                        "performance": {}
                    }
                
                # Get current prices for all assets
                asset_prices = await self._get_current_prices()
                
                total_value_idr = 0.0
                assets = []
                
                for entry in portfolio_entries:
                    asset_info = {
                        "currency": entry.currency,
                        "balance": entry.balance,
                        "locked_balance": entry.locked_balance
                    }
                    
                    if entry.currency == 'idr':
                        asset_info["value_idr"] = entry.balance
                        asset_info["price"] = 1.0
                    else:
                        pair_id = f"{entry.currency}_idr"
                        price = asset_prices.get(pair_id, 0.0)
                        asset_info["price"] = price
                        asset_info["value_idr"] = entry.balance * price
                    
                    total_value_idr += asset_info["value_idr"]
                    assets.append(asset_info)
                
                # Calculate allocation percentages
                allocation = {}
                for asset in assets:
                    if total_value_idr > 0:
                        allocation[asset["currency"]] = (asset["value_idr"] / total_value_idr) * 100
                    else:
                        allocation[asset["currency"]] = 0
                
                # Get performance data
                performance = await self._calculate_performance(user, assets)
                
                return {
                    "total_value_idr": total_value_idr,
                    "assets": assets,
                    "allocation": allocation,
                    "performance": performance,
                    "updated_at": datetime.now()
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to get portfolio summary", user_id=user.id, error=str(e))
            return {
                "total_value_idr": 0.0,
                "assets": [],
                "allocation": {},
                "performance": {}
            }
    
    async def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all available pairs"""
        try:
            # Get all tickers from Indodax
            ticker_data = await indodax_api.get_ticker_all()
            
            if not ticker_data or 'tickers' not in ticker_data:
                return {}
            
            prices = {}
            for pair_id, ticker in ticker_data['tickers'].items():
                prices[pair_id] = float(ticker.get('last', 0))
            
            return prices
            
        except Exception as e:
            logger.error("Failed to get current prices", error=str(e))
            return {}
    
    async def _calculate_performance(self, user: User, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            performance = {
                "total_pnl": 0.0,
                "total_pnl_percentage": 0.0,
                "daily_change": 0.0,
                "weekly_change": 0.0,
                "monthly_change": 0.0,
                "best_performer": None,
                "worst_performer": None
            }
            
            # Calculate PnL from trading history
            db = get_db()
            try:
                trades = db.query(Trade).filter(
                    Trade.user_id == user.id,
                    Trade.status == 'completed'
                ).all()
                
                total_investment = 0.0
                current_value = sum(asset["value_idr"] for asset in assets)
                
                for trade in trades:
                    if trade.type == 'buy':
                        total_investment += trade.total
                    else:  # sell
                        total_investment -= trade.total
                
                if total_investment > 0:
                    performance["total_pnl"] = current_value - total_investment
                    performance["total_pnl_percentage"] = (performance["total_pnl"] / total_investment) * 100
                
                # Calculate time-based changes (simplified)
                performance["daily_change"] = await self._calculate_period_change(user, 1)
                performance["weekly_change"] = await self._calculate_period_change(user, 7)
                performance["monthly_change"] = await self._calculate_period_change(user, 30)
                
                # Find best and worst performers
                asset_performances = []
                for asset in assets:
                    if asset["currency"] != 'idr':
                        asset_change = await self._calculate_asset_change(asset["currency"], 7)
                        asset_performances.append({
                            "currency": asset["currency"],
                            "change": asset_change
                        })
                
                if asset_performances:
                    best = max(asset_performances, key=lambda x: x["change"])
                    worst = min(asset_performances, key=lambda x: x["change"])
                    
                    performance["best_performer"] = best
                    performance["worst_performer"] = worst
                
                return performance
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to calculate performance", user_id=user.id, error=str(e))
            return {
                "total_pnl": 0.0,
                "total_pnl_percentage": 0.0,
                "daily_change": 0.0,
                "weekly_change": 0.0,
                "monthly_change": 0.0,
                "best_performer": None,
                "worst_performer": None
            }
    
    async def _calculate_period_change(self, user: User, days: int) -> float:
        """Calculate portfolio change over a specific period"""
        try:
            # This would compare current portfolio value with value N days ago
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            logger.error("Failed to calculate period change", user_id=user.id, days=days, error=str(e))
            return 0.0
    
    async def _calculate_asset_change(self, currency: str, days: int) -> float:
        """Calculate price change for an asset over a specific period"""
        try:
            pair_id = f"{currency}_idr"
            
            # Get current price
            ticker_data = await indodax_api.get_ticker(pair_id)
            if not ticker_data or 'ticker' not in ticker_data:
                return 0.0
            
            current_price = float(ticker_data['ticker']['last'])
            
            # Get historical price
            db = get_db()
            try:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                historical_price = db.query(PriceHistory).filter(
                    PriceHistory.pair_id == pair_id,
                    PriceHistory.timestamp >= cutoff_date
                ).order_by(PriceHistory.timestamp.asc()).first()
                
                if historical_price:
                    old_price = historical_price.close_price
                    if old_price > 0:
                        return ((current_price - old_price) / old_price) * 100
                
                return 0.0
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error("Failed to calculate asset change", currency=currency, days=days, error=str(e))
            return 0.0
    
    async def get_portfolio_diversification(self, user: User) -> Dict[str, Any]:
        """Analyze portfolio diversification"""
        try:
            portfolio_summary = await self.get_portfolio_summary(user)
            allocation = portfolio_summary.get("allocation", {})
            
            # Calculate diversification metrics
            diversification = {
                "asset_count": len([k for k, v in allocation.items() if v > 1]),  # Assets > 1%
                "concentration_risk": "low",
                "largest_position": 0.0,
                "recommendations": []
            }
            
            if allocation:
                largest_position = max(allocation.values())
                diversification["largest_position"] = largest_position
                
                # Determine concentration risk
                if largest_position > 70:
                    diversification["concentration_risk"] = "high"
                    diversification["recommendations"].append(
                        "Portfolio sangat terkonsentrasi. Pertimbangkan diversifikasi."
                    )
                elif largest_position > 50:
                    diversification["concentration_risk"] = "medium"
                    diversification["recommendations"].append(
                        "Portfolio cukup terkonsentrasi. Diversifikasi bisa mengurangi risiko."
                    )
                
                # Check if too many small positions
                small_positions = len([v for v in allocation.values() if 0 < v < 5])
                if small_positions > 5:
                    diversification["recommendations"].append(
                        "Terlalu banyak posisi kecil. Pertimbangkan konsolidasi."
                    )
                
                # Recommend rebalancing if needed
                if diversification["asset_count"] < 3 and portfolio_summary["total_value_idr"] > 1000000:
                    diversification["recommendations"].append(
                        "Pertimbangkan menambah aset untuk diversifikasi yang lebih baik."
                    )
            
            return diversification
            
        except Exception as e:
            logger.error("Failed to analyze diversification", user_id=user.id, error=str(e))
            return {
                "asset_count": 0,
                "concentration_risk": "unknown",
                "largest_position": 0.0,
                "recommendations": []
            }
    
    async def suggest_rebalancing(self, user: User) -> List[Dict[str, Any]]:
        """Suggest portfolio rebalancing actions"""
        try:
            portfolio_summary = await self.get_portfolio_summary(user)
            allocation = portfolio_summary.get("allocation", {})
            
            suggestions = []
            
            # Target allocation (simplified model)
            target_allocation = {
                "idr": 20,    # 20% cash
                "btc": 40,    # 40% Bitcoin  
                "eth": 25,    # 25% Ethereum
                "others": 15  # 15% other assets
            }
            
            # Calculate current allocation vs target
            current_btc = allocation.get("btc", 0)
            current_eth = allocation.get("eth", 0)
            current_idr = allocation.get("idr", 0)
            current_others = sum(v for k, v in allocation.items() if k not in ["btc", "eth", "idr"])
            
            # Generate suggestions based on differences
            if abs(current_btc - target_allocation["btc"]) > 10:
                if current_btc < target_allocation["btc"]:
                    suggestions.append({
                        "action": "buy",
                        "asset": "btc",
                        "reason": f"BTC allocation {current_btc:.1f}% below target {target_allocation['btc']}%"
                    })
                else:
                    suggestions.append({
                        "action": "sell",
                        "asset": "btc", 
                        "reason": f"BTC allocation {current_btc:.1f}% above target {target_allocation['btc']}%"
                    })
            
            if abs(current_eth - target_allocation["eth"]) > 10:
                if current_eth < target_allocation["eth"]:
                    suggestions.append({
                        "action": "buy",
                        "asset": "eth",
                        "reason": f"ETH allocation {current_eth:.1f}% below target {target_allocation['eth']}%"
                    })
                else:
                    suggestions.append({
                        "action": "sell",
                        "asset": "eth",
                        "reason": f"ETH allocation {current_eth:.1f}% above target {target_allocation['eth']}%"
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error("Failed to suggest rebalancing", user_id=user.id, error=str(e))
            return []
    
    async def get_portfolio_report(self, user: User) -> str:
        """Generate comprehensive portfolio report"""
        try:
            portfolio_summary = await self.get_portfolio_summary(user)
            diversification = await self.get_portfolio_diversification(user)
            rebalancing = await self.suggest_rebalancing(user)
            
            report = "📊 <b>Laporan Portfolio</b>\n\n"
            
            # Total value
            total_value = portfolio_summary.get("total_value_idr", 0)
            report += f"💰 <b>Total Nilai:</b> {format_currency(total_value)}\n\n"
            
            # Performance
            performance = portfolio_summary.get("performance", {})
            total_pnl = performance.get("total_pnl", 0)
            total_pnl_pct = performance.get("total_pnl_percentage", 0)
            
            pnl_emoji = "📈" if total_pnl >= 0 else "📉"
            report += f"{pnl_emoji} <b>P&L:</b> {format_currency(total_pnl)} ({total_pnl_pct:+.2f}%)\n"
            
            daily_change = performance.get("daily_change", 0)
            weekly_change = performance.get("weekly_change", 0)
            
            report += f"📅 <b>Perubahan Harian:</b> {daily_change:+.2f}%\n"
            report += f"📆 <b>Perubahan Mingguan:</b> {weekly_change:+.2f}%\n\n"
            
            # Asset allocation
            report += "<b>🔄 Alokasi Aset:</b>\n"
            allocation = portfolio_summary.get("allocation", {})
            
            for currency, percentage in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
                if percentage > 0.1:  # Only show assets > 0.1%
                    emoji = "💵" if currency == "idr" else "₿"
                    report += f"{emoji} {currency.upper()}: {percentage:.1f}%\n"
            
            # Diversification analysis
            report += f"\n<b>📊 Analisis Diversifikasi:</b>\n"
            report += f"• Jumlah Aset: {diversification['asset_count']}\n"
            report += f"• Risiko Konsentrasi: {diversification['concentration_risk'].upper()}\n"
            report += f"• Posisi Terbesar: {diversification['largest_position']:.1f}%\n"
            
            # Recommendations
            if diversification["recommendations"]:
                report += "\n<b>💡 Rekomendasi:</b>\n"
                for i, rec in enumerate(diversification["recommendations"][:3], 1):
                    report += f"{i}. {rec}\n"
            
            # Rebalancing suggestions
            if rebalancing:
                report += "\n<b>⚖️ Saran Rebalancing:</b>\n"
                for suggestion in rebalancing[:3]:
                    action_emoji = "🟢" if suggestion["action"] == "buy" else "🔴"
                    report += f"{action_emoji} {suggestion['action'].upper()} {suggestion['asset'].upper()}: {suggestion['reason']}\n"
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate portfolio report", user_id=user.id, error=str(e))
            return "❌ Gagal membuat laporan portfolio"

# Global portfolio manager instance
portfolio_manager = PortfolioManager()
