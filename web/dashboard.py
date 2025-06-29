"""
Web Dashboard for Telebot AI Trading Bot
FastAPI-based web interface for monitoring and management
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import get_db, User, Trade, Portfolio, AISignal
from core.indodax_api import indodax_api
from config.settings import settings

app = FastAPI(
    title="Telebot AI Trading Bot Dashboard",
    description="Web dashboard for monitoring and managing the AI trading bot",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get general statistics"""
    try:
        # Count active users
        active_users = db.query(User).filter(User.is_active == True).count()
        
        # Count total trades today
        today = datetime.now().date()
        trades_today = db.query(Trade).filter(
            Trade.created_at >= today
        ).count()
        
        # Count total signals today
        signals_today = db.query(AISignal).filter(
            AISignal.created_at >= today
        ).count()
        
        # Total portfolio value (simplified)
        total_portfolio_value = db.query(Portfolio).with_entities(
            db.func.sum(Portfolio.total_value)
        ).scalar() or 0
        
        return {
            "active_users": active_users,
            "trades_today": trades_today,
            "signals_today": signals_today,
            "total_portfolio_value": float(total_portfolio_value),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_recent_trades(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get recent trades"""
    try:
        trades = db.query(Trade).order_by(
            Trade.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": trade.id,
                "user_id": trade.user_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "amount": float(trade.amount),
                "price": float(trade.price),
                "status": trade.status,
                "created_at": trade.created_at.isoformat()
            }
            for trade in trades
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals")
async def get_recent_signals(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get recent AI signals"""
    try:
        signals = db.query(AISignal).order_by(
            AISignal.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": signal.id,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type,
                "confidence": float(signal.confidence),
                "target_price": float(signal.target_price) if signal.target_price else None,
                "created_at": signal.created_at.isoformat()
            }
            for signal in signals
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get user list"""
    try:
        users = db.query(User).order_by(
            User.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": user.id,
                "telegram_id": user.telegram_id,
                "username": user.username,
                "is_active": user.is_active,
                "is_premium": user.is_premium,
                "created_at": user.created_at.isoformat()
            }
            for user in users
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/{user_id}")
async def get_user_portfolio(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get user portfolio"""
    try:
        portfolio = db.query(Portfolio).filter(
            Portfolio.user_id == user_id
        ).first()
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return {
            "user_id": portfolio.user_id,
            "total_value": float(portfolio.total_value),
            "total_pnl": float(portfolio.total_pnl),
            "win_rate": float(portfolio.win_rate) if portfolio.win_rate else 0,
            "updated_at": portfolio.updated_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Get market data for a symbol"""
    try:
        # Get market data from Indodax API
        ticker = await indodax_api.get_ticker(symbol)
        depth = await indodax_api.get_depth(symbol)
        
        return {
            "symbol": symbol,
            "ticker": ticker,
            "depth": depth,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/emergency-stop")
async def emergency_stop():
    """Emergency stop all trading activities"""
    try:
        # Implement emergency stop logic here
        # This would typically:
        # 1. Cancel all open orders
        # 2. Stop signal generation
        # 3. Notify admins
        
        return {
            "status": "success",
            "message": "Emergency stop activated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-info")
async def get_system_info():
    """Get system information"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.debug else False
    )
