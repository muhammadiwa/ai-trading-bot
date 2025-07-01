"""
Enhanced Telegram Bot Features
Advanced AI-powered trading features for the crypto trading bot
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import pandas as pd
import numpy as np
import structlog

from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from core.database import get_db, User, Trade, Portfolio, AISignal, UserSettings
from core.indodax_api import IndodaxAPI
from ai.signal_generator import SignalGenerator
from core.backtester import Backtester, BacktestResult
from bot.utils import format_currency
from bot.messages import Messages

logger = structlog.get_logger(__name__)

class BacktestStates(StatesGroup):
    waiting_for_pair = State()
    waiting_for_strategy = State()
    waiting_for_period = State()
    waiting_for_parameters = State()

class SignalStates(StatesGroup):
    selecting_pair = State()
    configuring_filters = State()
    setting_alerts = State()

class EnhancedTradingFeatures:
    """Enhanced trading features with AI and backtesting"""
    
    def __init__(self, bot, signal_generator: SignalGenerator):
        self.bot = bot
        self.signal_generator = signal_generator
        self.backtester = Backtester()
        self.messages = Messages()
        self.api = IndodaxAPI()
        
        # Cache for performance
        self._signal_cache = {}
        self._backtest_cache = {}
        
    async def enhanced_signal_generation(self, message: Message, pair_id: Optional[str] = None) -> None:
        """Generate enhanced AI signals with multiple analysis methods"""
        try:
            user_id = message.from_user.id if message.from_user else 0
            
            # If no pair specified, show pair selection
            if not pair_id:
                await self._show_signal_pair_selection(message)
                return
                
            # Show processing message
            processing_msg = await message.answer(
                "🤖 <b>Menganalisis pasar dengan AI...</b>\n\n"
                f"📊 Pair: {pair_id.upper()}\n"
                "⏳ Mengumpulkan data real-time..."
            )
            
            # Generate comprehensive signal analysis
            signal_analysis = await self._generate_comprehensive_signal(pair_id)
            
            if not signal_analysis:
                await processing_msg.edit_text("❌ Gagal menganalisis sinyal. Coba lagi nanti.")
                return
                
            # Update processing message
            await processing_msg.edit_text(
                "🤖 <b>Menganalisis pasar dengan AI...</b>\n\n"
                f"📊 Pair: {pair_id.upper()}\n"
                "⏳ Menjalankan analisis teknikal..."
            )
            
            # Generate performance comparison with different strategies
            strategy_comparison = await self._compare_strategies(pair_id)
            
            # Update processing message
            await processing_msg.edit_text(
                "🤖 <b>Menganalisis pasar dengan AI...</b>\n\n"
                f"📊 Pair: {pair_id.upper()}\n"
                "⏳ Menyelesaikan analisis..."
            )
            
            # Format comprehensive signal report
            signal_report = await self._format_enhanced_signal_report(
                pair_id, signal_analysis, strategy_comparison
            )
            
            # Create enhanced keyboard with advanced options
            keyboard = self._create_enhanced_signal_keyboard(pair_id, signal_analysis)
            
            # Delete processing message and send final result
            await processing_msg.delete()
            await message.answer(signal_report, reply_markup=keyboard, parse_mode="HTML")
            
            # Cache signal for quick access
            self._signal_cache[f"{user_id}_{pair_id}"] = {
                "analysis": signal_analysis,
                "timestamp": datetime.now(),
                "strategies": strategy_comparison
            }
            
        except Exception as e:
            logger.error("Failed to generate enhanced signal", error=str(e))
            await message.answer("❌ Terjadi kesalahan saat menganalisis sinyal.")
    
    async def _generate_comprehensive_signal(self, pair_id: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive signal analysis using multiple AI methods"""
        try:
            # Generate primary AI signal
            primary_signal = await self.signal_generator.generate_signal(pair_id)
            
            if not primary_signal:
                return None
                
            # Get market data for additional analysis
            historical_data = await self.signal_generator._get_historical_data(pair_id, days=30)
            if historical_data is None or len(historical_data) < 20:
                return {"primary_signal": primary_signal, "market_data": None}
            
            # Calculate technical indicators
            indicators = self.signal_generator._calculate_technical_indicators(historical_data)
            
            # Perform market sentiment analysis
            market_sentiment = await self._analyze_market_sentiment(pair_id, historical_data)
            
            # Risk assessment
            risk_metrics = await self._calculate_risk_metrics(pair_id, historical_data, indicators)
            
            # Volume profile analysis
            volume_analysis = await self._analyze_volume_profile(historical_data)
            
            # Price action patterns
            pattern_analysis = await self._analyze_price_patterns(historical_data)
            
            return {
                "primary_signal": primary_signal,
                "indicators": indicators,
                "market_sentiment": market_sentiment,
                "risk_metrics": risk_metrics,
                "volume_analysis": volume_analysis,
                "pattern_analysis": pattern_analysis,
                "market_data": historical_data,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error("Failed to generate comprehensive signal", error=str(e))
            return None
    
    async def _analyze_market_sentiment(self, pair_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market sentiment using price action and volume"""
        try:
            if len(historical_data) < 20:
                return {"sentiment": "neutral", "confidence": 0.5, "factors": []}
            
            # Calculate recent price momentum
            recent_prices = historical_data['close'].tail(10)
            price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            
            # Volume trend analysis
            recent_volume = historical_data['volume'].tail(10)
            volume_trend = (recent_volume.mean() - historical_data['volume'].tail(20).head(10).mean()) / historical_data['volume'].tail(20).head(10).mean()
            
            # Volatility analysis
            returns = historical_data['close'].pct_change().tail(20)
            volatility = returns.std()
            
            # Fear & Greed indicators
            rsi_avg = self.signal_generator._get_current_indicators(
                {"rsi": historical_data['close'].rolling(14).apply(lambda x: self._calculate_rsi(x))}
            ).get("rsi", 50)
            
            # Determine sentiment
            sentiment_score = 0.0
            factors = []
            
            # Price momentum factor (40% weight)
            if price_momentum > 0.05:  # 5% up
                sentiment_score += 0.4
                factors.append("Strong upward price momentum")
            elif price_momentum < -0.05:  # 5% down
                sentiment_score -= 0.4
                factors.append("Strong downward price momentum")
            
            # Volume factor (30% weight)
            if volume_trend > 0.2:  # 20% volume increase
                sentiment_score += 0.3
                factors.append("Increasing volume supports price movement")
            elif volume_trend < -0.2:
                sentiment_score -= 0.3
                factors.append("Decreasing volume")
            
            # RSI factor (20% weight)
            if rsi_avg < 30:
                sentiment_score += 0.2
                factors.append("Oversold conditions (RSI)")
            elif rsi_avg > 70:
                sentiment_score -= 0.2
                factors.append("Overbought conditions (RSI)")
            
            # Volatility factor (10% weight)
            if volatility > 0.05:  # High volatility
                sentiment_score -= 0.1
                factors.append("High volatility indicates uncertainty")
            
            # Normalize sentiment score
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Determine sentiment label
            if sentiment_score > 0.3:
                sentiment = "bullish"
            elif sentiment_score < -0.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "confidence": abs(sentiment_score),
                "score": sentiment_score,
                "factors": factors,
                "metrics": {
                    "price_momentum": price_momentum,
                    "volume_trend": volume_trend,
                    "volatility": volatility,
                    "rsi": rsi_avg
                }
            }
            
        except Exception as e:
            logger.error("Failed to analyze market sentiment", error=str(e))
            return {"sentiment": "neutral", "confidence": 0.5, "factors": []}
    
    def _calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI for sentiment analysis"""
        try:
            delta = prices.diff()
            try:
                delta = pd.to_numeric(delta, errors='coerce')
                gain = delta.where(delta > 0, 0.0).mean()
                loss = (-delta.where(delta < 0, 0.0)).mean()
            except:
                # Fallback calculation
                gain = 0.0
                loss = 0.0
                for val in delta:
                    try:
                        val_float = float(str(val))
                        if val_float > 0:
                            gain += val_float
                        elif val_float < 0:
                            loss += abs(val_float)
                    except:
                        continue
                gain = gain / len(delta) if len(delta) > 0 else 0.0
                loss = loss / len(delta) if len(delta) > 0 else 0.0
            
            if loss == 0:
                return 100
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return 50  # neutral RSI
    
    async def _calculate_risk_metrics(self, pair_id: str, historical_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            if len(historical_data) < 20:
                return {"risk_level": "medium", "metrics": {}}
            
            # Price volatility
            returns = historical_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(365)  # Annualized
            
            # Value at Risk (VaR) - 5% confidence level
            var_5 = np.percentile(returns.dropna(), 5)
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Support/Resistance levels
            current_price = historical_data['close'].iloc[-1]
            recent_high = historical_data['high'].tail(20).max()
            recent_low = historical_data['low'].tail(20).min()
            
            # Distance from key levels
            distance_from_high = (recent_high - current_price) / current_price
            distance_from_low = (current_price - recent_low) / current_price
            
            # Risk score calculation
            risk_score = 0.0
            
            # Volatility risk (40% weight)
            if volatility > 0.8:  # Very high volatility
                risk_score += 0.4
            elif volatility > 0.4:  # High volatility
                risk_score += 0.2
            
            # Drawdown risk (30% weight)
            if abs(max_drawdown) > 0.3:  # More than 30% drawdown
                risk_score += 0.3
            elif abs(max_drawdown) > 0.15:  # More than 15% drawdown
                risk_score += 0.15
            
            # Position in range risk (20% weight)
            if distance_from_high < 0.05:  # Near resistance
                risk_score += 0.2
            elif distance_from_low < 0.05:  # Near support
                risk_score += 0.1
            
            # VaR risk (10% weight)
            if var_5 < -0.1:  # VaR worse than -10%
                risk_score += 0.1
            
            # Determine risk level
            if risk_score > 0.6:
                risk_level = "high"
            elif risk_score > 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "metrics": {
                    "volatility": volatility,
                    "var_5_percent": var_5,
                    "max_drawdown": max_drawdown,
                    "distance_from_high": distance_from_high,
                    "distance_from_low": distance_from_low,
                    "support_level": recent_low,
                    "resistance_level": recent_high
                }
            }
            
        except Exception as e:
            logger.error("Failed to calculate risk metrics", error=str(e))
            return {"risk_level": "medium", "metrics": {}}
    
    async def _analyze_volume_profile(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile and flow"""
        try:
            if 'volume' not in historical_data.columns or len(historical_data) < 10:
                return {"trend": "neutral", "strength": 0.5}
            
            # Recent volume trend
            recent_volume = historical_data['volume'].tail(10)
            previous_volume = historical_data['volume'].tail(20).head(10)
            
            volume_change = (recent_volume.mean() - previous_volume.mean()) / previous_volume.mean()
            
            # Volume-price relationship
            price_change = historical_data['close'].pct_change().tail(10)
            volume_normalized = (recent_volume - recent_volume.mean()) / recent_volume.std()
            
            # Calculate volume-price correlation
            try:
                correlation = np.corrcoef(price_change.dropna(), volume_normalized[-len(price_change.dropna()):])[0, 1]
            except:
                correlation = 0
            
            # On-Balance Volume trend
            obv = []
            obv_value = 0
            for i in range(len(historical_data)):
                if i > 0:
                    if historical_data['close'].iloc[i] > historical_data['close'].iloc[i-1]:
                        obv_value += historical_data['volume'].iloc[i]
                    elif historical_data['close'].iloc[i] < historical_data['close'].iloc[i-1]:
                        obv_value -= historical_data['volume'].iloc[i]
                obv.append(obv_value)
            
            obv_trend = "neutral"
            if len(obv) >= 10:
                recent_obv = obv[-5:]
                previous_obv = obv[-10:-5]
                if np.mean(recent_obv) > np.mean(previous_obv):
                    obv_trend = "bullish"
                elif np.mean(recent_obv) < np.mean(previous_obv):
                    obv_trend = "bearish"
            
            # Determine overall volume trend
            if volume_change > 0.2 and correlation > 0.3:
                trend = "strong_bullish"
                strength = 0.8
            elif volume_change > 0.2 and correlation < -0.3:
                trend = "strong_bearish"
                strength = 0.8
            elif volume_change > 0.1:
                trend = "increasing"
                strength = 0.6
            elif volume_change < -0.1:
                trend = "decreasing"
                strength = 0.4
            else:
                trend = "neutral"
                strength = 0.5
            
            return {
                "trend": trend,
                "strength": strength,
                "volume_change": volume_change,
                "price_volume_correlation": correlation,
                "obv_trend": obv_trend,
                "average_volume": recent_volume.mean()
            }
            
        except Exception as e:
            logger.error("Failed to analyze volume profile", error=str(e))
            return {"trend": "neutral", "strength": 0.5}
    
    async def _analyze_price_patterns(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action patterns"""
        try:
            if len(historical_data) < 20:
                return {"patterns": [], "trend": "neutral"}
            
            patterns = []
            highs = historical_data['high'].tail(20)
            lows = historical_data['low'].tail(20)
            closes = historical_data['close'].tail(20)
            
            # Support and resistance levels
            recent_highs = highs.tail(5)
            recent_lows = lows.tail(5)
            
            # Pattern: Higher highs and higher lows (uptrend)
            if recent_highs.iloc[-1] > recent_highs.iloc[0] and recent_lows.iloc[-1] > recent_lows.iloc[0]:
                patterns.append({
                    "name": "Higher Highs & Higher Lows",
                    "type": "bullish",
                    "strength": 0.7
                })
            
            # Pattern: Lower highs and lower lows (downtrend)
            elif recent_highs.iloc[-1] < recent_highs.iloc[0] and recent_lows.iloc[-1] < recent_lows.iloc[0]:
                patterns.append({
                    "name": "Lower Highs & Lower Lows",
                    "type": "bearish",
                    "strength": 0.7
                })
            
            # Pattern: Double top/bottom
            high_peaks = []
            low_valleys = []
            
            for i in range(1, len(highs)-1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    high_peaks.append((i, highs.iloc[i]))
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    low_valleys.append((i, lows.iloc[i]))
            
            # Check for double top
            if len(high_peaks) >= 2:
                last_two_peaks = high_peaks[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:  # Within 2%
                    patterns.append({
                        "name": "Double Top",
                        "type": "bearish",
                        "strength": 0.6
                    })
            
            # Check for double bottom
            if len(low_valleys) >= 2:
                last_two_valleys = low_valleys[-2:]
                if abs(last_two_valleys[0][1] - last_two_valleys[1][1]) / last_two_valleys[0][1] < 0.02:  # Within 2%
                    patterns.append({
                        "name": "Double Bottom",
                        "type": "bullish",
                        "strength": 0.6
                    })
            
            # Pattern: Consolidation/sideways
            price_range = (highs.max() - lows.min()) / closes.mean()
            if price_range < 0.05:  # Less than 5% range
                patterns.append({
                    "name": "Consolidation",
                    "type": "neutral",
                    "strength": 0.5
                })
            
            # Overall trend determination
            if any(p["type"] == "bullish" for p in patterns):
                trend = "bullish"
            elif any(p["type"] == "bearish" for p in patterns):
                trend = "bearish"
            else:
                trend = "neutral"
            
            return {
                "patterns": patterns,
                "trend": trend,
                "support_level": lows.min(),
                "resistance_level": highs.max()
            }
            
        except Exception as e:
            logger.error("Failed to analyze price patterns", error=str(e))
            return {"patterns": [], "trend": "neutral"}
    
    async def _compare_strategies(self, pair_id: str) -> Dict[str, Any]:
        """Compare different trading strategies performance"""
        try:
            # Quick backtest for different strategies
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            strategies = ["ai_signals", "buy_hold", "dca"]
            strategy_results = {}
            
            for strategy in strategies:
                try:
                    result = await self.backtester.run_backtest(
                        pair_id=pair_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_balance=1000000,
                        strategy=strategy,
                        min_confidence=0.6
                    )
                    
                    if result:
                        strategy_results[strategy] = {
                            "return": result.total_return_percent,
                            "sharpe": result.sharpe_ratio,
                            "max_drawdown": result.max_drawdown,
                            "win_rate": result.win_rate,
                            "total_trades": result.total_trades
                        }
                    else:
                        strategy_results[strategy] = None
                        
                except Exception as e:
                    logger.error(f"Failed to backtest {strategy}", error=str(e))
                    strategy_results[strategy] = None
            
            # Find best performing strategy
            best_strategy = None
            best_return = float('-inf')
            
            for strategy, result in strategy_results.items():
                if result and result["return"] > best_return:
                    best_return = result["return"]
                    best_strategy = strategy
            
            return {
                "results": strategy_results,
                "best_strategy": best_strategy,
                "best_return": best_return
            }
            
        except Exception as e:
            logger.error("Failed to compare strategies", error=str(e))
            return {"results": {}, "best_strategy": None}
    
    async def _format_enhanced_signal_report(self, pair_id: str, analysis: Dict[str, Any], 
                                           strategy_comparison: Dict[str, Any]) -> str:
        """Format comprehensive signal analysis report"""
        try:
            primary_signal = analysis.get("primary_signal")
            if not primary_signal:
                return "❌ Tidak dapat menghasilkan analisis sinyal."
            
            # Extract signal type and confidence
            signal_type = getattr(primary_signal, 'signal_type', 'hold')
            confidence = getattr(primary_signal, 'confidence', 0.5)
            
            # Get current price from indicators
            indicators = analysis.get("indicators", {})
            current_indicators = self.signal_generator._get_current_indicators(indicators) if indicators else {}
            current_price = current_indicators.get('close', 0)
            
            # Format signal emoji and strength
            if signal_type == "buy":
                signal_emoji = "🟢"
                signal_text = "BELI"
            elif signal_type == "sell":
                signal_emoji = "🔴"
                signal_text = "JUAL"
            else:
                signal_emoji = "🟡"
                signal_text = "TAHAN"
            
            # Confidence level
            if confidence >= 0.8:
                confidence_text = "Sangat Tinggi"
            elif confidence >= 0.65:
                confidence_text = "Tinggi"
            elif confidence >= 0.5:
                confidence_text = "Sedang"
            else:
                confidence_text = "Rendah"
            
            # Market sentiment
            sentiment_data = analysis.get("market_sentiment", {})
            sentiment = sentiment_data.get("sentiment", "neutral")
            sentiment_confidence = sentiment_data.get("confidence", 0.5)
            
            sentiment_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(sentiment, "➡️")
            
            # Risk assessment
            risk_data = analysis.get("risk_metrics", {})
            risk_level = risk_data.get("risk_level", "medium")
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "🟡")
            
            # Volume analysis
            volume_data = analysis.get("volume_analysis", {})
            volume_trend = volume_data.get("trend", "neutral")
            
            # Price patterns
            pattern_data = analysis.get("pattern_analysis", {})
            patterns = pattern_data.get("patterns", [])
            pattern_trend = pattern_data.get("trend", "neutral")
            
            # Strategy comparison
            best_strategy = strategy_comparison.get("best_strategy")
            best_return = strategy_comparison.get("best_return", 0)
            
            # Technical indicators summary
            tech_summary = []
            if 'rsi' in current_indicators:
                rsi = current_indicators['rsi']
                if rsi < 30:
                    tech_summary.append("RSI: Oversold")
                elif rsi > 70:
                    tech_summary.append("RSI: Overbought")
                else:
                    tech_summary.append(f"RSI: {rsi:.1f}")
            
            if 'macd' in current_indicators and 'macd_signal' in current_indicators:
                macd = current_indicators['macd']
                macd_signal = current_indicators['macd_signal']
                if macd > macd_signal:
                    tech_summary.append("MACD: Bullish")
                else:
                    tech_summary.append("MACD: Bearish")
            
            # Format main report
            report = f"""
🤖 <b>ANALISIS AI ADVANCED</b>

📊 <b>Pair:</b> {pair_id.upper()}
💰 <b>Harga:</b> {format_currency(current_price) if current_price else 'N/A'} IDR

{signal_emoji} <b>SINYAL: {signal_text}</b>
🎯 <b>Confidence:</b> {confidence:.1%} ({confidence_text})

📈 <b>ANALISIS PASAR</b>
• Sentimen: {sentiment_emoji} {sentiment.title()} ({sentiment_confidence:.1%})
• Risk Level: {risk_emoji} {risk_level.title()}
• Volume Trend: {volume_trend.replace('_', ' ').title()}
• Pattern: {pattern_trend.title()}

🔍 <b>INDIKATOR TEKNIKAL</b>
{chr(10).join([f"• {indicator}" for indicator in tech_summary[:3]])}

📋 <b>POLA HARGA</b>
{chr(10).join([f"• {pattern['name']}: {pattern['type'].title()}" for pattern in patterns[:2]]) if patterns else "• Tidak ada pola signifikan"}

⚡ <b>STRATEGI TERBAIK (30 hari)</b>
• {best_strategy.replace('_', ' ').title() if best_strategy else 'N/A'}: {best_return:.2f}%

🎯 <b>REKOMENDASI</b>
"""
            
            # Add specific recommendations based on analysis
            if signal_type == "buy" and confidence >= 0.7:
                report += "• Peluang beli yang baik dengan confidence tinggi\n"
                if sentiment == "bullish":
                    report += "• Didukung sentimen pasar bullish\n"
                if risk_level == "low":
                    report += "• Risk rendah untuk entry\n"
            elif signal_type == "sell" and confidence >= 0.7:
                report += "• Pertimbangkan untuk take profit\n"
                if sentiment == "bearish":
                    report += "• Sentimen pasar mendukung aksi jual\n"
            else:
                report += "• Tunggu sinyal yang lebih kuat\n"
                report += "• Monitor perkembangan pasar\n"
            
            report += f"\n⚠️ <i>Analisis berdasarkan data {datetime.now().strftime('%d/%m/%Y %H:%M')} WIB</i>"
            
            return report
            
        except Exception as e:
            logger.error("Failed to format signal report", error=str(e))
            return "❌ Gagal memformat laporan analisis."
    
    def _create_enhanced_signal_keyboard(self, pair_id: str, analysis: Dict[str, Any]) -> InlineKeyboardMarkup:
        """Create enhanced keyboard for signal actions"""
        primary_signal = analysis.get("primary_signal")
        signal_type = getattr(primary_signal, 'signal_type', 'hold') if primary_signal else 'hold'
        
        keyboard = [
            [
                InlineKeyboardButton(
                    text="🔄 Refresh Signal",
                    callback_data=f"signal_refresh_{pair_id}"
                ),
                InlineKeyboardButton(
                    text="📊 Backtest",
                    callback_data=f"backtest_quick_{pair_id}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="📈 Technical Analysis",
                    callback_data=f"signal_technical_{pair_id}"
                ),
                InlineKeyboardButton(
                    text="⚡ Auto Trade",
                    callback_data=f"signal_auto_{pair_id}_{signal_type}"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔔 Set Alert",
                    callback_data=f"signal_alert_{pair_id}"
                ),
                InlineKeyboardButton(
                    text="📋 Strategy Compare",
                    callback_data=f"signal_compare_{pair_id}"
                )
            ]
        ]
        
        # Add trade button if signal is actionable
        if signal_type in ["buy", "sell"]:
            keyboard.append([
                InlineKeyboardButton(
                    text=f"💹 {signal_type.upper()}",
                    callback_data=f"trade_{pair_id}_{signal_type}"
                )
            ])
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard)
    
    async def _show_signal_pair_selection(self, message: Message) -> None:
        """Show pair selection for signal analysis"""
        try:
            # Get popular pairs from API
            pairs_data = await self.api.get_pairs()
            popular_pairs = []
            
            # Select most popular pairs
            priority_pairs = ['btc_idr', 'eth_idr', 'usdt_idr', 'bnb_idr', 'ada_idr', 'sol_idr']
            
            for pair_id in priority_pairs:
                for pair in pairs_data:
                    if pair.get('ticker_id') == pair_id:
                        popular_pairs.append(pair)
                        break
            
            # Create keyboard with popular pairs
            keyboard = []
            for i in range(0, len(popular_pairs), 2):
                row = []
                for j in range(2):
                    if i + j < len(popular_pairs):
                        pair = popular_pairs[i + j]
                        symbol = pair['ticker_id'].replace('_idr', '').upper()
                        row.append(InlineKeyboardButton(
                            text=f"{symbol}",
                            callback_data=f"signal_pair_{pair['ticker_id']}"
                        ))
                keyboard.append(row)
            
            # Add "Other" option
            keyboard.append([
                InlineKeyboardButton(
                    text="🔍 Pair Lainnya",
                    callback_data="signal_pair_other"
                )
            ])
            
            await message.answer(
                "📊 <b>PILIH PAIR UNTUK ANALISIS</b>\n\n"
                "Pilih cryptocurrency yang ingin dianalisis:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard)
            )
            
        except Exception as e:
            logger.error("Failed to show pair selection", error=str(e))
            await message.answer("❌ Gagal menampilkan pilihan pair.")
    
    async def quick_backtest(self, message: Message, pair_id: str, strategy: str = "ai_signals") -> None:
        """Run quick backtest and show results"""
        try:
            # Show processing message
            processing_msg = await message.answer(
                f"🔄 <b>Quick Backtest</b>\n\n"
                f"📊 Pair: {pair_id.upper()}\n"
                f"📈 Strategy: {strategy.replace('_', ' ').title()}\n"
                "⏳ Running analysis..."
            )
            
            # Run 7-day backtest
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            result = await self.backtester.run_backtest(
                pair_id=pair_id,
                start_date=start_date,
                end_date=end_date,
                initial_balance=1000000,
                strategy=strategy,
                min_confidence=0.65
            )
            
            if not result:
                await processing_msg.edit_text("❌ Gagal menjalankan backtest.")
                return
            
            # Format quick results
            return_color = "🟢" if result.total_return_percent > 0 else "🔴"
            
            quick_result = f"""
📊 <b>QUICK BACKTEST (7 HARI)</b>

📈 <b>Strategy:</b> {strategy.replace('_', ' ').title()}
💰 <b>Pair:</b> {pair_id.upper()}

{return_color} <b>Return: {result.total_return_percent:.2f}%</b>
📊 <b>Sharpe Ratio:</b> {result.sharpe_ratio:.2f}
📉 <b>Max Drawdown:</b> {result.max_drawdown:.2f}%
🎯 <b>Win Rate:</b> {result.win_rate:.1f}%
🔄 <b>Total Trades:</b> {result.total_trades}

💸 <b>Balance:</b>
• Initial: {format_currency(result.initial_balance)} IDR
• Final: {format_currency(result.final_balance)} IDR
• P&L: {format_currency(result.total_return)} IDR

⚠️ <i>Backtest 7 hari terakhir</i>
"""
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📊 Full Backtest",
                        callback_data=f"backtest_full_{pair_id}_{strategy}"
                    ),
                    InlineKeyboardButton(
                        text="⚡ Apply Strategy",
                        callback_data=f"backtest_apply_{pair_id}_{strategy}"
                    )
                ]
            ])
            
            await processing_msg.edit_text(quick_result, reply_markup=keyboard)
            
        except Exception as e:
            logger.error("Failed to run quick backtest", error=str(e))
            await message.answer("❌ Gagal menjalankan quick backtest.")
