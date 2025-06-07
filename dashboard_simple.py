"""
Simple Test Dashboard for AI Trading Bot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="AI Trading Bot Dashboard 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    }
    
    .status-online {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Trading Bot Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Navigation
        page = st.selectbox(
            "📊 Select Page:",
            ["🏠 Dashboard", "📈 Live Trading", "🎯 Market Analysis", 
             "💼 Portfolio", "🔄 Backtesting", "⚙️ Settings"]
        )
        
        st.markdown("---")
        
        # Bot Status
        st.subheader("🤖 Bot Status")
        is_running = st.checkbox("Bot Running")
        
        if is_running:
            st.markdown('<div class="metric-card status-online">🟢 RUNNING</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card status-offline">🔴 STOPPED</div>', unsafe_allow_html=True)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            st.button("🚀 Start")
        with col2:
            st.button("🛑 Stop")
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("💰 Account Info")
        st.metric("IDR Balance", "Rp 5,000,000")
        st.metric("BTC Balance", "0.00125")
        st.metric("ETH Balance", "0.025")
    
    # Main content
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

def show_dashboard():
    """Dashboard overview"""
    st.header("🏠 Trading Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Portfolio Value", "Rp 12,500,000", "5.2%")
    
    with col2:
        st.metric("💰 Today's P&L", "Rp 650,000", "Rp 125,000")
    
    with col3:
        st.metric("🎯 Success Rate", "78.5%", "2.1%")
    
    with col4:
        st.metric("🔄 Trades Today", "12", "3")
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Price Chart")
        show_price_chart()
    
    with col2:
        st.subheader("🎯 Current Signals")
        show_signals()
    
    # Recent trades
    st.subheader("📋 Recent Trading Activity")
    show_recent_trades()

def show_price_chart():
    """Show sample price chart"""
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    prices = np.random.normal(680000000, 10000000, 100).cumsum() / 100 + 680000000
    
    # Create candlestick data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices
    })
    
    fig = go.Figure(data=go.Candlestick(
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
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_signals():
    """Show current trading signals"""
    st.markdown("**🎯 Current Signal: BUY**")
    st.success("🟢 Strong Buy Signal")
    
    st.metric("₿ BTC Price", "Rp 685,420,000", "2.3%")
    
    st.write("**📊 Technical Indicators:**")
    st.write("• RSI: 65.2 (Neutral)")
    st.write("• MACD: Buy Signal")
    st.write("• SMA: Above 20-day")
    
    st.write("**🤖 AI Prediction:**")
    st.write("• Next Hour: Rp 687,500,000")
    st.write("• Confidence: 78%")

def show_recent_trades():
    """Show recent trading activity"""
    trades = pd.DataFrame({
        'Time': ['14:30:15', '13:25:42', '12:18:33', '11:45:12'],
        'Pair': ['BTC/IDR', 'ETH/IDR', 'BTC/IDR', 'ETH/IDR'],
        'Action': ['BUY', 'SELL', 'BUY', 'BUY'],
        'Amount': ['0.00125', '0.0425', '0.00089', '0.0312'],
        'Price': ['Rp 685,420,000', 'Rp 32,750,000', 'Rp 682,150,000', 'Rp 32,500,000'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success'],
        'P&L': ['+Rp 12,500', '+Rp 8,750', '+Rp 5,200', '+Rp 3,100']
    })
    
    st.dataframe(trades, use_container_width=True, hide_index=True)

def show_live_trading():
    """Live trading interface"""
    st.header("📈 Live Trading Interface")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎮 Trading Controls")
        
        pair = st.selectbox("Trading Pair", ["BTC/IDR", "ETH/IDR"])
        amount = st.number_input("Amount (IDR)", min_value=10000, value=100000)
        
        col_buy, col_sell = st.columns(2)
        
        with col_buy:
            if st.button("🟢 BUY", use_container_width=True):
                st.success("✅ Buy order placed!")
        
        with col_sell:
            if st.button("🔴 SELL", use_container_width=True):
                st.success("✅ Sell order placed!")
    
    with col2:
        st.subheader("📊 Live Chart")
        show_price_chart()

def show_market_analysis():
    """Market analysis page"""
    st.header("🎯 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Technical Indicators")
        
        # RSI gauge
        rsi_value = 65
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rsi_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "RSI"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}]}
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🤖 AI Analysis")
        
        st.metric("Predicted Price (1h)", "Rp 687,500,000", "0.3%")
        st.metric("Confidence Level", "78%")
        st.success("📈 Direction: BULLISH")

def show_portfolio():
    """Portfolio page"""
    st.header("💼 Portfolio Management")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Value", "Rp 12,500,000", "5.2%")
    
    with col2:
        st.metric("Available Balance", "Rp 4,500,000")
    
    with col3:
        st.metric("Invested Amount", "Rp 8,000,000")
    
    with col4:
        st.metric("Unrealized P&L", "Rp 1,250,000", "15.6%")
    
    # Asset allocation pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 Asset Allocation")
        
        assets = ['IDR', 'Bitcoin', 'Ethereum']
        values = [36, 45, 19]
        
        fig = px.pie(values=values, names=assets, title="Portfolio Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Portfolio Performance")
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        values = np.random.normal(100, 5, 30).cumsum() + 10000000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', fill='tonexty'))
        fig.update_layout(title="Portfolio Growth", xaxis_title="Date", yaxis_title="Value (IDR)")
        
        st.plotly_chart(fig, use_container_width=True)

def show_backtesting():
    """Backtesting page"""
    st.header("🔄 Strategy Backtesting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy = st.selectbox("Strategy", ["MA Cross", "RSI", "MACD"])
        initial_capital = st.number_input("Initial Capital", value=1000000)
    
    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        end_date = st.date_input("End Date", datetime.now())
    
    with col3:
        stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
        take_profit = st.slider("Take Profit (%)", 5, 50, 15)
    
    if st.button("🚀 Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            import time
            time.sleep(2)
            
            st.success("✅ Backtest completed!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", "15.8%")
            
            with col2:
                st.metric("Win Rate", "68%")
            
            with col3:
                st.metric("Max Drawdown", "-8.2%")
            
            with col4:
                st.metric("Sharpe Ratio", "1.45")

def show_settings():
    """Settings page"""
    st.header("⚙️ Bot Configuration")
    
    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input("Indodax API Key", type="password")
            telegram_token = st.text_input("Telegram Bot Token", type="password")
        
        with col2:
            secret_key = st.text_input("Indodax Secret Key", type="password")
            chat_id = st.text_input("Telegram Chat ID")
    
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
    
    if st.button("💾 Save Configuration", use_container_width=True):
        st.success("✅ Configuration saved successfully!")

if __name__ == "__main__":
    main()
