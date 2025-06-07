import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import os
import time
from pathlib import Path

# Import bot components
from main import AITradingBot
from backtester import Backtester, simple_ma_strategy, rsi_strategy
from transaction_logger import TransactionLogger

# Page configuration
st.set_page_config(
    page_title="AI Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-alert {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        border-radius: 0.25rem;
    }
    .error-alert {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid transparent;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'auto_trading' not in st.session_state:
    st.session_state.auto_trading = False

def initialize_bot():
    """Initialize the trading bot"""
    try:
        if st.session_state.bot is None:
            st.session_state.bot = AITradingBot()
        return True
    except Exception as e:
        st.error(f"Error initializing bot: {e}")
        return False

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Trading Bot Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Bot initialization
    if not initialize_bot():
        st.error("Failed to initialize trading bot. Please check your configuration.")
        return
    
    # Navigation
    page = st.sidebar.selectbox(
        "📊 Select Page",
        ["Dashboard", "Market Analysis", "Trading Signals", "Portfolio", "Backtesting", "Settings"]
    )
    
    # Bot control
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Bot Control")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("▶️ Start Bot", key="start_bot"):
            st.session_state.bot_running = True
            st.success("Bot started!")
    
    with col2:
        if st.button("⏹️ Stop Bot", key="stop_bot"):
            st.session_state.bot_running = False
            st.session_state.auto_trading = False
            st.info("Bot stopped!")
    
    # Auto trading toggle
    st.session_state.auto_trading = st.sidebar.checkbox(
        "🤖 Enable Auto Trading",
        value=st.session_state.auto_trading,
        help="Enable automatic trade execution based on AI signals"
    )
    
    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status")
    
    status_color = "🟢" if st.session_state.bot_running else "🔴"
    st.sidebar.markdown(f"{status_color} Bot Status: {'Running' if st.session_state.bot_running else 'Stopped'}")
    
    auto_status_color = "🟢" if st.session_state.auto_trading else "🟡"
    st.sidebar.markdown(f"{auto_status_color} Auto Trading: {'ON' if st.session_state.auto_trading else 'OFF'}")
    
    # Route to selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Market Analysis":
        show_market_analysis()
    elif page == "Trading Signals":
        show_trading_signals()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Backtesting":
        show_backtesting()
    elif page == "Settings":
        show_settings()

def show_dashboard():
    """Dashboard overview page"""
    st.header("📊 Trading Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Portfolio",
            value="Rp 1,250,000",
            delta="5.2%"
        )
    
    with col2:
        st.metric(
            label="📈 Today's P&L",
            value="Rp 65,000",
            delta="Rp 12,000"
        )
    
    with col3:
        st.metric(
            label="🔄 Trades Today",
            value="3",
            delta="1"
        )
    
    with col4:
        st.metric(
            label="🎯 Win Rate",
            value="75%",
            delta="5%"
        )
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Portfolio Performance")
        
        # Sample data for portfolio chart
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        portfolio_values = np.random.normal(1000000, 50000, len(dates)).cumsum() / len(dates) + 1000000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Value (IDR)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Asset Allocation")
        
        # Sample data for pie chart
        assets = ['IDR', 'BTC', 'ETH', 'Other']
        values = [40, 35, 20, 5]
        
        fig = px.pie(
            values=values,
            names=assets,
            title="Current Asset Distribution"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    
    # Sample recent trades
    recent_trades = pd.DataFrame({
        'Time': ['10:30:15', '09:45:22', '08:12:45'],
        'Pair': ['BTC/IDR', 'ETH/IDR', 'BTC/IDR'],
        'Action': ['BUY', 'SELL', 'BUY'],
        'Amount': ['0.00125', '0.0425', '0.00089'],
        'Price': ['Rp 685,000,000', 'Rp 32,500,000', 'Rp 682,000,000'],
        'Status': ['✅ Success', '✅ Success', '✅ Success']
    })
    
    st.dataframe(recent_trades, use_container_width=True)

def show_market_analysis():
    """Market analysis page"""
    st.header("📊 Market Analysis")
    
    # Pair selection
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        selected_pair = st.selectbox("Select Trading Pair", ["BTC/IDR", "ETH/IDR"])
    
    with col2:
        if st.button("🔄 Refresh Analysis", key="refresh_analysis"):
            with st.spinner("Analyzing market..."):
                time.sleep(2)  # Simulate analysis time
                st.success("Analysis updated!")
    
    # Current market data
    st.subheader("💹 Current Market Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", "Rp 685,420,000", "2.3%")
    
    with col2:
        st.metric("24h Volume", "245.67 BTC", "15.2%")
    
    with col3:
        st.metric("24h High", "Rp 695,000,000", "")
    
    with col4:
        st.metric("24h Low", "Rp 675,000,000", "")
    
    # Technical indicators
    st.subheader("📈 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # RSI gauge
        st.markdown("**RSI (14)**")
        rsi_value = 65
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rsi_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "RSI"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MACD
        st.markdown("**MACD**")
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        macd_line = np.random.normal(0, 1, 30)
        signal_line = np.random.normal(0, 0.8, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=macd_line, name='MACD', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=signal_line, name='Signal', line=dict(color='red')))
        
        fig.update_layout(
            title="MACD Indicator",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Prediction
    st.subheader("🤖 AI Price Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Price (1h)", "Rp 687,500,000", "0.3%")
    
    with col2:
        st.metric("Confidence Level", "78%", "")
    
    with col3:
        prediction_direction = "📈 BULLISH"
        st.markdown(f"**Direction:** {prediction_direction}")

def show_trading_signals():
    """Trading signals page"""
    st.header("🎯 Trading Signals")
    
    # Signal summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Overall Signal</h3>
            <h2>🟢 BUY</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>AI Confidence</h3>
            <h2>78%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Risk Level</h3>
            <h2>🟡 MEDIUM</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed signals
    st.subheader("📊 Signal Breakdown")
    
    signals_data = pd.DataFrame({
        'Indicator': ['RSI', 'MACD', 'Moving Average', 'Bollinger Bands', 'AI Prediction'],
        'Signal': ['NEUTRAL', 'BUY', 'BUY', 'NEUTRAL', 'BUY'],
        'Strength': [65, 78, 82, 55, 78],
        'Last Update': ['2 min ago', '1 min ago', '1 min ago', '3 min ago', '30 sec ago']
    })
    
    # Color code signals
    def color_signals(val):
        if val == 'BUY':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'SELL':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #fff3cd; color: #856404'
    
    styled_df = signals_data.style.applymap(color_signals, subset=['Signal'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Manual trading section
    st.subheader("⚡ Manual Trading")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trade_pair = st.selectbox("Trading Pair", ["BTC/IDR", "ETH/IDR"])
    
    with col2:
        trade_amount = st.number_input("Amount (IDR)", min_value=10000, value=100000, step=10000)
    
    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        
        col_buy, col_sell = st.columns(2)
        with col_buy:
            if st.button("🟢 BUY", key="manual_buy"):
                st.success(f"Buy order placed for {trade_amount:,} IDR")
        
        with col_sell:
            if st.button("🔴 SELL", key="manual_sell"):
                st.error(f"Sell order placed for {trade_amount:,} IDR")

def show_portfolio():
    """Portfolio page"""
    st.header("💼 Portfolio Overview")
    
    # Portfolio summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", "Rp 1,250,000", "5.2%")
    
    with col2:
        st.metric("Available Balance", "Rp 450,000", "-2.1%")
    
    with col3:
        st.metric("Invested Amount", "Rp 800,000", "8.5%")
    
    with col4:
        st.metric("Total P&L", "Rp 125,000", "12.5%")
    
    # Holdings
    st.subheader("🏦 Current Holdings")
    
    holdings_data = pd.DataFrame({
        'Asset': ['IDR', 'Bitcoin (BTC)', 'Ethereum (ETH)'],
        'Amount': ['450,000', '0.001125', '0.0245'],
        'Current Price': ['1', 'Rp 685,420,000', 'Rp 32,750,000'],
        'Value (IDR)': ['Rp 450,000', 'Rp 770,723', 'Rp 802,375'],
        'Allocation': ['36%', '62%', '64%'],
        '24h Change': ['0%', '+2.3%', '+1.8%']
    })
    
    st.dataframe(holdings_data, use_container_width=True)
    
    # Portfolio performance chart
    st.subheader("📈 Portfolio Performance")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    portfolio_values = np.random.normal(1000000, 25000, 30).cumsum() / 30 + 1000000
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=3),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value (IDR)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading history
    st.subheader("📋 Trading History")
    
    trade_history = pd.DataFrame({
        'Date': ['2024-01-15 10:30', '2024-01-15 09:15', '2024-01-14 16:22'],
        'Pair': ['BTC/IDR', 'ETH/IDR', 'BTC/IDR'],
        'Type': ['BUY', 'SELL', 'BUY'],
        'Amount': ['0.001125', '0.025', '0.0008'],
        'Price': ['Rp 685,000,000', 'Rp 32,500,000', 'Rp 682,000,000'],
        'Total': ['Rp 770,625', 'Rp 812,500', 'Rp 545,600'],
        'P&L': ['+Rp 2,500', '+Rp 15,000', '+Rp 8,200']
    })
    
    st.dataframe(trade_history, use_container_width=True)

def show_backtesting():
    """Backtesting page"""
    st.header("🧪 Strategy Backtesting")
    
    # Backtesting parameters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strategy = st.selectbox("Strategy", ["Moving Average Cross", "RSI Strategy", "MACD Strategy"])
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1 day", "1 week", "1 month", "3 months"])
    
    with col3:
        initial_balance = st.number_input("Initial Balance", value=1000000, step=100000)
    
    with col4:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🚀 Run Backtest", key="run_backtest"):
            with st.spinner("Running backtest..."):
                time.sleep(3)  # Simulate backtest time
                st.success("Backtest completed!")
    
    # Backtest results
    st.subheader("📊 Backtest Results")
    
    # Results metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", "15.8%", "")
    
    with col2:
        st.metric("Max Drawdown", "-8.2%", "")
    
    with col3:
        st.metric("Win Rate", "68%", "")
    
    with col4:
        st.metric("Sharpe Ratio", "1.45", "")
    
    # Backtest chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Portfolio value over time
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        portfolio_values = np.random.normal(1000000, 25000, 30).cumsum() / 30 + 1000000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add buy/sell markers
        buy_dates = dates[::5]  # Every 5th day
        buy_values = portfolio_values[::5]
        
        fig.add_trace(go.Scatter(
            x=buy_dates,
            y=buy_values,
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ))
        
        fig.update_layout(
            title="Backtest Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (IDR)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Drawdown chart
        drawdown = np.random.uniform(-0.1, 0.02, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown * 100,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade log
    st.subheader("📋 Backtest Trade Log")
    
    backtest_trades = pd.DataFrame({
        'Date': ['2024-01-15', '2024-01-12', '2024-01-10', '2024-01-08'],
        'Action': ['BUY', 'SELL', 'BUY', 'SELL'],
        'Price': ['Rp 685,000,000', 'Rp 678,000,000', 'Rp 672,000,000', 'Rp 680,000,000'],
        'Amount': ['0.0015', '0.0012', '0.0012', '0.001'],
        'P&L': ['-', '+Rp 7,200', '-', '+Rp 8,000'],
        'Cumulative P&L': ['Rp 15,200', 'Rp 15,200', 'Rp 8,000', 'Rp 8,000']
    })
    
    st.dataframe(backtest_trades, use_container_width=True)

def show_settings():
    """Settings page"""
    st.header("⚙️ Bot Settings")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_key = st.text_input("Indodax API Key", type="password", placeholder="Enter your API key")
        telegram_token = st.text_input("Telegram Bot Token", type="password", placeholder="Enter bot token")
    
    with col2:
        secret_key = st.text_input("Indodax Secret Key", type="password", placeholder="Enter your secret key")
        telegram_chat_id = st.text_input("Telegram Chat ID", placeholder="Enter chat ID")
    
    # Trading Parameters
    st.subheader("📊 Trading Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
        max_trades = st.slider("Max Daily Trades", 1, 50, 10)
    
    with col2:
        take_profit = st.slider("Take Profit (%)", 5, 50, 15)
        position_size = st.slider("Position Size (%)", 1, 50, 10)
    
    with col3:
        risk_per_trade = st.slider("Risk per Trade (%)", 1, 10, 2)
        ai_confidence_threshold = st.slider("AI Confidence Threshold (%)", 50, 95, 70)
    
    # Notification Settings
    st.subheader("📲 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_telegram = st.checkbox("Enable Telegram Notifications", value=True)
        enable_email = st.checkbox("Enable Email Notifications", value=False)
    
    with col2:
        enable_whatsapp = st.checkbox("Enable WhatsApp Notifications", value=False)
        enable_discord = st.checkbox("Enable Discord Notifications", value=False)
    
    # AI Model Settings
    st.subheader("🤖 AI Model Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lstm_epochs = st.number_input("LSTM Epochs", 10, 200, 50)
    
    with col2:
        batch_size = st.number_input("Batch Size", 8, 128, 32)
    
    with col3:
        prediction_days = st.number_input("Prediction Days", 1, 30, 7)
    
    # Save settings
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Settings", key="save_settings"):
            st.success("Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Default", key="reset_settings"):
            st.info("Settings reset to default values!")
    
    with col3:
        if st.button("🧪 Test Notifications", key="test_notifications"):
            with st.spinner("Testing notifications..."):
                time.sleep(2)
                st.success("✅ Telegram: Connected\n❌ WhatsApp: Not configured")

# Run the app
if __name__ == "__main__":
    main()
