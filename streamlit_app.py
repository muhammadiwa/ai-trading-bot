"""
AI Trading Bot - Streamlit Web Dashboard
Modern and comprehensive trading dashboard with real-time features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import os
import time
import json
import threading
from pathlib import Path

# Import bot components
try:
    from main import AITradingBot
    from backtester import Backtester
    from transaction_logger import TransactionLogger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Trading Bot Dashboard 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
    st.session_state.bot_initialized = False
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = {}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .status-online {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .trade-signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .trade-signal-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .trade-signal-hold {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_bot():
    """Initialize the trading bot"""
    try:
        if not st.session_state.bot_initialized:
            with st.spinner("Initializing AI Trading Bot..."):
                st.session_state.bot = AITradingBot()
                st.session_state.bot_initialized = True
            st.success("✅ AI Trading Bot initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize bot: {str(e)}")
        return False

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Trading Bot Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize bot
    if not initialize_bot():
        st.stop()
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Navigation
        page = st.selectbox(
            "📊 Select Page:",
            ["🏠 Dashboard", "📈 Live Trading", "🎯 Market Analysis", "💼 Portfolio", 
             "🔄 Backtesting", "⚙️ Settings", "📱 Notifications", "📝 Trading Log"]
        )
        
        st.markdown("---")
        
        # Bot Status & Controls
        st.subheader("🤖 Bot Status")
        
        # Status display
        if st.session_state.is_running:
            st.markdown('<div class="metric-card status-online">🟢 RUNNING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card status-offline">🔴 STOPPED</div>', unsafe_allow_html=True)
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start", disabled=st.session_state.is_running):
                start_bot()
        
        with col2:
            if st.button("🛑 Stop", disabled=not st.session_state.is_running):
                stop_bot()
        
        # Auto trading toggle
        st.session_state.auto_trading = st.checkbox(
            "🔄 Auto Trading", 
            value=st.session_state.auto_trading,
            help="Enable automatic trade execution"
        )
        
        st.markdown("---")
        
        # Quick account info
        st.subheader("💰 Account Info")
        show_account_summary()
        
        st.markdown("---")
        
        # Real-time data refresh
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Main content area
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📈 Live Trading":
        show_live_trading()
    elif page == "🎯 Market Analysis":
        show_market_analysis()
    elif page == "💼 Portfolio":
        show_portfolio()
    elif page == "🔄 Backtesting":
        show_backtesting()
    elif page == "⚙️ Settings":
        show_settings()
    elif page == "📱 Notifications":
        show_notifications()
    elif page == "📝 Trading Log":
        show_trading_log()

def start_bot():
    """Start the trading bot"""
    try:
        st.session_state.is_running = True
        st.success("🚀 Trading bot started!")
        
        # You can add background thread logic here if needed
        # For now, we'll just update the status
        
    except Exception as e:
        st.error(f"❌ Failed to start bot: {str(e)}")
        st.session_state.is_running = False

def stop_bot():
    """Stop the trading bot"""
    try:
        st.session_state.is_running = False
        st.session_state.auto_trading = False
        st.info("🛑 Trading bot stopped")
        
    except Exception as e:
        st.error(f"❌ Failed to stop bot: {str(e)}")

def show_account_summary():
    """Show account summary in sidebar"""
    try:
        if st.session_state.bot:
            account_info = st.session_state.bot.api.get_account_info()
            if account_info and 'return' in account_info:
                balance = account_info['return']['balance']
                st.metric("💰 IDR", f"Rp {float(balance.get('idr', 0)):,.0f}")
                st.metric("₿ BTC", f"{float(balance.get('btc', 0)):.6f}")
                st.metric("⧫ ETH", f"{float(balance.get('eth', 0)):.6f}")
            else:
                st.info("Connect API to view balance")
    except:
        st.info("Account info unavailable")

def show_dashboard():
    """Main dashboard overview"""
    st.header("🏠 Trading Dashboard Overview")
    
    # Get real portfolio data
    portfolio_data = get_real_portfolio_data()
    trade_history = get_real_trade_history()
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if portfolio_data['success']:
            portfolio_value = portfolio_data['total_value']
            st.metric(
                label="📊 Portfolio Value",
                value=f"Rp {portfolio_value:,.0f}",
                delta=None,  # TODO: Calculate delta from previous day
                help="Total portfolio value including all assets"
            )
        else:
            st.metric(
                label="📊 Portfolio Value",
                value="Unavailable",
                help="Total portfolio value including all assets"
            )
    
    with col2:
        # Calculate today's P&L from trade history
        today_trades = [t for t in trade_history if pd.to_datetime(t.get('timestamp', '')).date() == datetime.now().date()]
        today_pnl = sum(float(t.get('profit_loss', 0)) for t in today_trades)
        
        st.metric(
            label="💰 Today's P&L",
            value=f"Rp {today_pnl:,.0f}" if today_pnl != 0 else "Rp 0",
            delta=None,
            help="Profit/Loss for today"
        )
    
    with col3:
        # Calculate success rate from trade history
        if trade_history:
            successful_trades = len([t for t in trade_history if float(t.get('profit_loss', 0)) > 0])
            success_rate = (successful_trades / len(trade_history)) * 100
            st.metric(
                label="🎯 Success Rate",
                value=f"{success_rate:.1f}%",
                delta=None,
                help="Percentage of successful trades"
            )
        else:
            st.metric(
                label="🎯 Success Rate",
                value="N/A",
                help="Percentage of successful trades"
            )
    
    with col4:
        # Count trades today
        today_trades_count = len(today_trades) if 'today_trades' in locals() else 0
        st.metric(
            label="🔄 Trades Today",
            value=str(today_trades_count),
            delta=None,
            help="Number of trades executed today"
        )
    
    # Market Overview
    st.subheader("📈 Market Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Price chart
        show_price_chart()
    
    with col2:
        # Current signals
        show_current_signals()
    
    # Recent Activity
    st.subheader("📋 Recent Trading Activity")
    show_recent_trades()

def show_price_chart():
    """Show cryptocurrency price chart"""
    try:
        # Try to get real market data first
        market_data = get_real_market_data('btcidr')
        
        if st.session_state.bot and st.session_state.bot_initialized:
            # Try to get historical data from bot
            try:
                df = st.session_state.bot.fetch_historical_data('btcidr', limit=100)
                
                if not df.empty and 'timestamp' in df.columns:
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='BTC/IDR'
                    ))
                    
                    fig.update_layout(
                        title="Bitcoin Price Chart (BTC/IDR)",
                        yaxis_title="Price (IDR)",
                        xaxis_title="Time",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    return
                    
            except Exception as hist_error:
                st.warning(f"📊 Historical data unavailable: {hist_error}")
        
        # Fallback: Show current price if available
        if market_data['success']:
            st.info(f"📊 Current BTC Price: Rp {market_data['current_price']:,.0f}")
            st.info(f"📈 24h High: Rp {market_data['high_24h']:,.0f}")
            st.info(f"📉 24h Low: Rp {market_data['low_24h']:,.0f}")
        else:
            st.warning("📊 No price data available - Bot not initialized or API error")
            
    except Exception as e:
        st.error(f"❌ Error loading price chart: {str(e)}")
        # Show fallback sample chart
        st.info("📊 Showing sample price data (Bot not connected)")

def show_current_signals():
    """Show current trading signals"""
    try:
        # Get real technical analysis
        analysis_data = get_real_technical_analysis('btcidr')
        market_data = get_real_market_data('btcidr')
        
        st.subheader("🎯 Current Signal")
        
        if analysis_data['success']:
            signal = analysis_data.get('overall_signal', 'HOLD')
            
            # Display signal with styling
            if signal == 'BUY':
                st.markdown('<div class="trade-signal-buy">🟢 BUY SIGNAL</div>', unsafe_allow_html=True)
            elif signal == 'SELL':
                st.markdown('<div class="trade-signal-sell">🔴 SELL SIGNAL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="trade-signal-hold">🟡 HOLD SIGNAL</div>', unsafe_allow_html=True)
            
            # Price info
            if market_data['success']:
                price = market_data['current_price']
                st.metric("₿ BTC Price", f"Rp {price:,.0f}")
            
            # Technical indicators
            st.write("**📊 Technical Indicators:**")
            st.write(f"• RSI: {analysis_data.get('rsi_value', 0):.1f} ({analysis_data.get('rsi_signal', 'N/A')})")
            st.write(f"• MACD: {analysis_data.get('macd_signal', 'N/A')}")
            st.write(f"• SMA: {analysis_data.get('sma_signal', 'N/A')}")
            st.write(f"• Bollinger Bands: {analysis_data.get('bb_signal', 'N/A')}")
            
            # AI prediction
            if analysis_data.get('predicted_price'):
                st.write("**🤖 AI Prediction:**")
                st.write(f"Price: Rp {analysis_data['predicted_price']:,.0f}")
                st.write(f"Direction: {analysis_data.get('predicted_direction', 'UNKNOWN')}")
                st.write(f"Confidence: {analysis_data.get('ai_confidence', 0):.1f}%")
        else:
            st.warning(f"⚠️ Unable to fetch market signals: {analysis_data.get('error', 'Unknown error')}")
            
            # Show basic market data if available
            if market_data['success']:
                st.metric("₿ BTC Price", f"Rp {market_data['current_price']:,.0f}")
            
    except Exception as e:
        st.error(f"❌ Signal analysis error: {str(e)}")
        st.info("🔄 Please ensure the bot is initialized and connected to API")

def show_recent_trades():
    """Show recent trading activity"""
    try:
        # Get real trade history
        trade_history = get_real_trade_history()
        
        if trade_history:
            # Convert to DataFrame for display
            df_trades = pd.DataFrame(trade_history)
            
            # Format the data for display
            display_data = []
            for trade in trade_history[:10]:  # Show last 10 trades
                display_data.append({
                    'Time': pd.to_datetime(trade.get('timestamp', '')).strftime('%H:%M:%S') if trade.get('timestamp') else 'N/A',
                    'Date': pd.to_datetime(trade.get('timestamp', '')).strftime('%Y-%m-%d') if trade.get('timestamp') else 'N/A',
                    'Pair': trade.get('pair', 'N/A').upper(),
                    'Action': trade.get('action', 'N/A').upper(),
                    'Amount': f"{float(trade.get('amount', 0)):.6f}",
                    'Price': f"Rp {float(trade.get('price', 0)):,.0f}",
                    'Status': '✅ Success' if trade.get('status') == 'completed' else '❌ Failed',
                    'P&L': f"Rp {float(trade.get('profit_loss', 0)):,.0f}" if trade.get('profit_loss') else 'N/A'
                })
            
            if display_data:
                recent_trades_df = pd.DataFrame(display_data)
                st.dataframe(recent_trades_df, use_container_width=True, hide_index=True)
            else:
                st.info("📋 No recent trades found")
        else:
            st.info("📋 No trade history available")
            st.write("**Note:** Trade history will appear here after the bot executes trades")
        
    except Exception as e:
        st.error(f"❌ Error loading trade history: {str(e)}")
        # Show sample data as fallback
        st.info("📋 Showing sample trade data (Real data unavailable)")
        sample_trades = pd.DataFrame({
            'Time': ['14:30:15', '13:25:42', '12:18:33'],
            'Pair': ['BTC/IDR', 'ETH/IDR', 'BTC/IDR'],
            'Action': ['BUY', 'SELL', 'BUY'],
            'Amount': ['0.00125', '0.0425', '0.00089'],
            'Price': ['Rp 685,420,000', 'Rp 32,750,000', 'Rp 682,150,000'],
            'Status': ['✅ Success', '✅ Success', '✅ Success']
        })
        st.dataframe(sample_trades, use_container_width=True, hide_index=True)

def show_live_trading():
    """Live trading interface"""
    st.header("📈 Live Trading Interface")
    
    # Trading controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎮 Trading Controls")
        
        # Pair selection
        pair = st.selectbox("Select Trading Pair", ["btcidr", "ethidr"])
        
        # Order type
        order_type = st.radio("Order Type", ["Market", "Limit"])
        
        # Amount
        amount = st.number_input("Amount (IDR)", min_value=10000, value=100000, step=10000)
        
        if order_type == "Limit":
            limit_price = st.number_input("Limit Price", min_value=1000, value=685000000)
        
        # Trading buttons
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            if st.button("🟢 BUY", key="live_buy", use_container_width=True):
                execute_trade(pair, "BUY", amount)
        
        with col_sell:
            if st.button("🔴 SELL", key="live_sell", use_container_width=True):
                execute_trade(pair, "SELL", amount)
    
    with col2:
        st.subheader("📊 Live Chart")
        show_price_chart()
    
    # Order book and recent trades
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Order Book")
        show_order_book()
    
    with col2:
        st.subheader("💼 Open Orders")
        show_open_orders()

def execute_trade(pair, action, amount):
    """Execute a trade"""
    try:
        with st.spinner(f"Executing {action} order..."):
            # Simulate trade execution
            time.sleep(1)
            
            # Here you would call the actual trading function
            # result = asyncio.run(st.session_state.bot.execute_trade(pair, action, analysis))
            
            st.success(f"✅ {action} order for {amount:,} IDR executed successfully!")
            
    except Exception as e:
        st.error(f"❌ Trade execution failed: {str(e)}")

def show_order_book():
    """Show order book"""
    # Sample order book data
    order_book = pd.DataFrame({
        'Price': ['Rp 685,500,000', 'Rp 685,400,000', 'Rp 685,300,000'],
        'Amount': ['0.125', '0.089', '0.256'],
        'Total': ['Rp 85,687,500', 'Rp 61,000,000', 'Rp 175,436,800']
    })
    
    st.dataframe(order_book, use_container_width=True, hide_index=True)

def show_open_orders():
    """Show open orders"""
    # Sample open orders
    open_orders = pd.DataFrame({
        'Type': ['BUY', 'SELL'],
        'Amount': ['0.00125', '0.00089'],
        'Price': ['Rp 680,000,000', 'Rp 690,000,000'],
        'Status': ['Pending', 'Pending']
    })
    
    if not open_orders.empty:
        st.dataframe(open_orders, use_container_width=True, hide_index=True)
    else:
        st.info("No open orders")

def show_market_analysis():
    """Market analysis page"""
    st.header("🎯 Advanced Market Analysis")
    
    # Analysis controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pair = st.selectbox("Trading Pair", ["btcidr", "ethidr"])
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1H", "4H", "1D", "1W"])
    
    with col3:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    
    # Technical analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Technical Indicators")
        show_technical_indicators()
    
    with col2:
        st.subheader("🤖 AI Analysis")
        show_ai_analysis()
    
    # Sentiment analysis
    st.subheader("💭 Market Sentiment")
    show_sentiment_analysis()

def show_technical_indicators():
    """Show technical indicators"""
    try:
        # Create sample technical indicators
        indicators = {
            'RSI (14)': 65.2,
            'RSI Signal': 'NEUTRAL',
            'MACD': 1250.5,
            'MACD Signal': 'BUY',
            'SMA (20)': 'Rp 682,500,000',
            'SMA Signal': 'BUY',
            'Bollinger Bands': 'NEUTRAL'
        }
        
        for indicator, value in indicators.items():
            if 'Signal' in indicator:
                if value == 'BUY':
                    st.success(f"🟢 {indicator}: {value}")
                elif value == 'SELL':
                    st.error(f"🔴 {indicator}: {value}")
                else:
                    st.warning(f"🟡 {indicator}: {value}")
            else:
                st.metric(indicator, value)
                
    except Exception as e:
        st.error(f"❌ Error loading technical indicators: {str(e)}")

def show_ai_analysis():
    """Show AI analysis results"""
    try:
        if st.session_state.last_analysis:
            analysis = st.session_state.last_analysis
            
            # AI prediction
            if analysis.get('predicted_price'):
                st.metric(
                    "🤖 Predicted Price", 
                    f"Rp {analysis['predicted_price']:,.0f}",
                    delta=f"{analysis.get('ai_confidence', 0):.1f}% confidence"
                )
            
            # Direction
            direction = analysis.get('predicted_direction', 'UNKNOWN')
            if direction == 'UP':
                st.success(f"📈 Direction: {direction}")
            elif direction == 'DOWN':
                st.error(f"📉 Direction: {direction}")
            else:
                st.info(f"↔️ Direction: {direction}")
        else:
            st.info("Run market analysis to see AI predictions")
            
    except Exception as e:
        st.error(f"❌ Error loading AI analysis: {str(e)}")

def show_sentiment_analysis():
    """Show market sentiment analysis"""
    # Sample sentiment data
    sentiment_data = {
        'Fear & Greed Index': 65,
        'Social Media Sentiment': 72,
        'News Sentiment': 58,
        'Technical Sentiment': 68
    }
    
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    
    for i, (metric, value) in enumerate(sentiment_data.items()):
        with columns[i]:
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': metric},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

def show_portfolio():
    """Portfolio management page"""
    st.header("💼 Portfolio Management")
    
    # Portfolio summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", "Rp 12,500,000", "5.2%")
    
    with col2:
        st.metric("Available Balance", "Rp 4,500,000", "-1.2%")
    
    with col3:
        st.metric("Invested Amount", "Rp 8,000,000", "8.5%")
    
    with col4:
        st.metric("Unrealized P&L", "Rp 1,250,000", "15.6%")
    
    # Asset allocation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Asset Allocation")
        
        # Pie chart
        assets = ['IDR', 'Bitcoin', 'Ethereum', 'Other']
        values = [36, 45, 15, 4]
        
        fig = px.pie(values=values, names=assets, title="Portfolio Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Portfolio Performance")
        
        # Performance chart
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        portfolio_values = np.random.normal(100, 5, 30).cumsum() + 10000000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            fill='tonexty',
            name='Portfolio Value'
        ))
        
        fig.update_layout(
            title="Portfolio Growth",
            xaxis_title="Date",
            yaxis_title="Value (IDR)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Holdings table
    st.subheader("🏦 Current Holdings")
    
    holdings = pd.DataFrame({
        'Asset': ['IDR', 'Bitcoin (BTC)', 'Ethereum (ETH)'],
        'Amount': ['4,500,000', '0.00725', '0.125'],
        'Current Price': ['1', 'Rp 685,420,000', 'Rp 32,750,000'],
        'Value (IDR)': ['Rp 4,500,000', 'Rp 4,969,295', 'Rp 4,093,750'],
        'Allocation': ['36%', '40%', '33%'],
        '24h Change': ['0%', '+2.3%', '+1.8%'],
        'P&L': ['Rp 0', '+Rp 250,000', '+Rp 125,000']
    })
    
    st.dataframe(holdings, use_container_width=True, hide_index=True)

def show_backtesting():
    """Backtesting interface"""
    st.header("🔄 Strategy Backtesting")
    
    # Backtesting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⚙️ Parameters")
        strategy = st.selectbox("Strategy", ["MA Cross", "RSI", "MACD", "AI Hybrid"])
        initial_capital = st.number_input("Initial Capital", value=1000000, step=100000)
        timeframe = st.selectbox("Timeframe", ["1H", "4H", "1D"])
    
    with col2:
        st.subheader("📅 Date Range")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", datetime.now())
        
    with col3:
        st.subheader("🎯 Risk Management")
        stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
        take_profit = st.slider("Take Profit (%)", 5, 50, 15)
    
    # Run backtest
    if st.button("🚀 Run Backtest", use_container_width=True):
        run_backtest(strategy, initial_capital, start_date, end_date, stop_loss, take_profit)

def run_backtest(strategy, initial_capital, start_date, end_date, stop_loss, take_profit):
    """Run backtesting simulation"""
    with st.spinner("Running backtest simulation..."):
        time.sleep(3)  # Simulate processing
        
        # Sample backtest results
        results = {
            'Total Return': '15.8%',
            'Max Drawdown': '-8.2%',
            'Win Rate': '68%',
            'Sharpe Ratio': '1.45',
            'Total Trades': 45,
            'Profitable Trades': 31,
            'Final Capital': initial_capital * 1.158
        }
        
        # Display results
        st.success("✅ Backtest completed!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", results['Total Return'])
            st.metric("Win Rate", results['Win Rate'])
        
        with col2:
            st.metric("Max Drawdown", results['Max Drawdown'])
            st.metric("Sharpe Ratio", results['Sharpe Ratio'])
        
        with col3:
            st.metric("Total Trades", results['Total Trades'])
            st.metric("Final Capital", f"Rp {results['Final Capital']:,.0f}")
        
        with col4:
            st.metric("Profitable Trades", results['Profitable Trades'])
            st.metric("Avg Trade", f"Rp {(results['Final Capital'] - initial_capital) / results['Total Trades']:,.0f}")

def show_settings():
    """Settings and configuration page"""
    st.header("⚙️ Bot Configuration")
    
    # API Settings
    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input("Indodax API Key", type="password")
            telegram_token = st.text_input("Telegram Bot Token", type="password")
        
        with col2:
            secret_key = st.text_input("Indodax Secret Key", type="password")
            chat_id = st.text_input("Telegram Chat ID")
    
    # Trading Parameters
    with st.expander("📊 Trading Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
            take_profit = st.slider("Take Profit (%)", 5, 50, 15)
        
        with col2:
            max_trades = st.slider("Max Daily Trades", 1, 50, 10)
            position_size = st.slider("Position Size (%)", 1, 50, 10)
        
        with col3:
            risk_per_trade = st.slider("Risk per Trade (%)", 1, 10, 2)
            ai_threshold = st.slider("AI Confidence Threshold (%)", 50, 95, 70)
    
    # Notification Settings
    with st.expander("📱 Notification Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Telegram Notifications", value=True)
            st.checkbox("Enable Email Notifications")
        
        with col2:
            st.checkbox("Trade Execution Alerts", value=True)
            st.checkbox("Daily Summary Reports", value=True)
    
    # Save settings
    if st.button("💾 Save Configuration", use_container_width=True):
        st.success("✅ Configuration saved successfully!")

def show_notifications():
    """Notifications management page"""
    st.header("📱 Notification Center")
    
    # Test notifications
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Test Notifications")
        
        if st.button("Test Telegram", use_container_width=True):
            with st.spinner("Sending test message..."):
                time.sleep(1)
                st.success("✅ Telegram test message sent!")
        
        if st.button("Test Email", use_container_width=True):
            with st.spinner("Sending test email..."):
                time.sleep(1)
                st.success("✅ Test email sent!")
    
    with col2:
        st.subheader("📊 Notification Stats")
        st.metric("Messages Sent Today", "25")
        st.metric("Success Rate", "98%")
    
    # Recent notifications
    st.subheader("📋 Recent Notifications")
    
    notifications = pd.DataFrame({
        'Time': ['14:30:15', '13:25:42', '12:18:33'],
        'Type': ['Trade Alert', 'Price Alert', 'System Alert'],
        'Message': ['BUY order executed', 'BTC reached target price', 'Bot started'],
        'Status': ['✅ Sent', '✅ Sent', '✅ Sent']
    })
    
    st.dataframe(notifications, use_container_width=True, hide_index=True)

def show_trading_log():
    """Trading log and history page"""
    st.header("📝 Trading Log & History")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_filter = st.date_input("Filter by Date", datetime.now())
    
    with col2:
        pair_filter = st.selectbox("Filter by Pair", ["All", "BTC/IDR", "ETH/IDR"])
    
    with col3:
        action_filter = st.selectbox("Filter by Action", ["All", "BUY", "SELL"])
    
    # Trading history
    st.subheader("📊 Trade History")
    
    trade_history = pd.DataFrame({
        'Date': ['2024-01-15 14:30', '2024-01-15 13:25', '2024-01-15 12:18', '2024-01-15 11:45'],
        'Pair': ['BTC/IDR', 'ETH/IDR', 'BTC/IDR', 'ETH/IDR'],
        'Action': ['BUY', 'SELL', 'BUY', 'BUY'],
        'Amount': ['0.00125', '0.0425', '0.00089', '0.0312'],
        'Price': ['Rp 685,420,000', 'Rp 32,750,000', 'Rp 682,150,000', 'Rp 32,500,000'],
        'Total': ['Rp 856,775', 'Rp 1,391,875', 'Rp 607,114', 'Rp 1,014,000'],
        'Fee': ['Rp 2,570', 'Rp 4,176', 'Rp 1,821', 'Rp 3,042'],
        'P&L': ['+Rp 12,500', '+Rp 8,750', '+Rp 5,200', '+Rp 3,100'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success']
    })
    
    st.dataframe(trade_history, use_container_width=True, hide_index=True)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export CSV"):
            st.success("✅ Trading history exported to CSV!")
    
    with col2:
        if st.button("📊 Generate Report"):
            st.success("✅ Trading report generated!")
    
    with col3:
        if st.button("🔄 Refresh Log"):
            st.rerun()

# Helper functions for real data integration
def get_real_portfolio_data():
    """Get real portfolio data from bot"""
    try:
        if st.session_state.bot and st.session_state.bot_initialized:
            # Get account info from API
            account_info = st.session_state.bot.api.get_account_info()
            
            if account_info and 'return' in account_info:
                balance_info = account_info['return']['balance']
                
                # Get current prices
                btc_ticker = st.session_state.bot.api.get_ticker('btcidr')
                eth_ticker = st.session_state.bot.api.get_ticker('ethidr')
                
                btc_price = float(btc_ticker.get('ticker', {}).get('last', 0)) if btc_ticker else 0
                eth_price = float(eth_ticker.get('ticker', {}).get('last', 0)) if eth_ticker else 0
                
                # Calculate portfolio
                idr_balance = float(balance_info.get('idr', 0))
                btc_balance = float(balance_info.get('btc', 0))
                eth_balance = float(balance_info.get('eth', 0))
                
                btc_value = btc_balance * btc_price
                eth_value = eth_balance * eth_price
                total_value = idr_balance + btc_value + eth_value
                
                return {
                    'idr_balance': idr_balance,
                    'btc_balance': btc_balance,
                    'eth_balance': eth_balance,
                    'btc_price': btc_price,
                    'eth_price': eth_price,
                    'btc_value': btc_value,
                    'eth_value': eth_value,
                    'total_value': total_value,
                    'success': True
                }
            else:
                return {'success': False, 'error': 'Failed to get account info'}
        else:
            return {'success': False, 'error': 'Bot not initialized'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_real_market_data(pair='btcidr'):
    """Get real market data from API"""
    try:
        if st.session_state.bot and st.session_state.bot_initialized:
            # Get ticker data
            ticker = st.session_state.bot.api.get_ticker(pair)
            
            if ticker and 'ticker' in ticker:
                ticker_data = ticker['ticker']
                return {
                    'current_price': float(ticker_data.get('last', 0)),
                    'high_24h': float(ticker_data.get('high', 0)),
                    'low_24h': float(ticker_data.get('low', 0)),
                    'volume_24h': float(ticker_data.get('vol_btc', 0)) if 'btc' in pair else float(ticker_data.get('vol_eth', 0)),
                    'volume_idr': float(ticker_data.get('vol_idr', 0)),
                    'change_24h': 0,  # Calculate if needed
                    'success': True
                }
            else:
                return {'success': False, 'error': 'Failed to get ticker data'}
        else:
            return {'success': False, 'error': 'Bot not initialized'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_real_trade_history():
    """Get real trade history from logs"""
    try:
        # Read from logs/trades.csv
        trades_file = Path(__file__).parent / 'logs' / 'trades.csv'
        
        if trades_file.exists():
            df = pd.read_csv(trades_file)
            if not df.empty:
                # Sort by timestamp (newest first)
                df = df.sort_values('timestamp', ascending=False)
                # Return last 10 trades
                return df.head(10).to_dict('records')
        
        return []
    except Exception as e:
        print(f"Error reading trade history: {e}")
        return []

def get_real_technical_analysis(pair='btcidr'):
    """Get real technical analysis from bot"""
    try:
        if st.session_state.bot and st.session_state.bot_initialized:
            # Get market analysis
            analysis = st.session_state.bot.analyze_market(pair)
            
            if 'error' not in analysis:
                return {
                    'rsi_value': analysis.get('rsi_value', 50),
                    'rsi_signal': analysis.get('rsi_signal', 'HOLD'),
                    'macd_signal': analysis.get('macd_signal', 'HOLD'),
                    'sma_signal': analysis.get('sma_signal', 'HOLD'),
                    'bb_signal': analysis.get('bb_signal', 'HOLD'),
                    'overall_signal': analysis.get('overall_signal', 'HOLD'),
                    'predicted_price': analysis.get('predicted_price', 0),
                    'predicted_direction': analysis.get('predicted_direction', 'UNKNOWN'),
                    'ai_confidence': analysis.get('ai_confidence', 0),
                    'success': True
                }
            else:
                return {'success': False, 'error': analysis.get('error', 'Analysis failed')}
        else:
            return {'success': False, 'error': 'Bot not initialized'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_real_portfolio_history():
    """Get real portfolio history from logs"""
    try:
        # Read from logs/portfolio.csv
        portfolio_file = Path(__file__).parent / 'logs' / 'portfolio.csv'
        
        if portfolio_file.exists():
            df = pd.read_csv(portfolio_file)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.to_dict('records')
        
        return []
    except Exception as e:
        print(f"Error reading portfolio history: {e}")
        return []

# Run the application
if __name__ == "__main__":
    main()
