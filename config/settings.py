import os
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    """Application settings loaded from environment variables"""
    
    # Telegram Configuration
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_webhook_url: str = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    telegram_admin_ids: List[int] = [int(x.strip()) for x in os.getenv("TELEGRAM_ADMIN_IDS", "123456789").split(",") if x.strip()]
    
    # Indodax API Configuration
    indodax_api_key: str = os.getenv("INDODAX_API_KEY", "")
    indodax_secret_key: str = os.getenv("INDODAX_SECRET_KEY", "")
    indodax_base_url: str = os.getenv("INDODAX_BASE_URL", "https://indodax.com")
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./trading_bot.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # AI/ML Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_path: str = os.getenv("MODEL_PATH", "./ai/models/")
    training_data_path: str = os.getenv("TRAINING_DATA_PATH", "./data/training/")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Trading Configuration
    default_trading_amount: float = float(os.getenv("DEFAULT_TRADING_AMOUNT", "100000"))
    max_daily_trades: int = int(os.getenv("MAX_DAILY_TRADES", "10"))
    stop_loss_percentage: float = float(os.getenv("STOP_LOSS_PERCENTAGE", "5.0"))
    take_profit_percentage: float = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "10.0"))
    
    # Risk Management
    max_portfolio_risk: float = float(os.getenv("MAX_PORTFOLIO_RISK", "20.0"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "30.0"))
    min_balance_idr: float = float(os.getenv("MIN_BALANCE_IDR", "50000"))
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Rate Limiting
    api_rate_limit: int = int(os.getenv("API_RATE_LIMIT", "180"))
    api_rate_window: int = int(os.getenv("API_RATE_WINDOW", "60"))
    
    # External APIs
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    
    # Monitoring
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    
    class Config:
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """Validate that all required settings are present"""
    required_settings = [
        "telegram_bot_token",
        "indodax_api_key", 
        "indodax_secret_key",
        "secret_key"
    ]
    
    missing_settings = []
    for setting in required_settings:
        if not getattr(settings, setting):
            missing_settings.append(setting.upper())
    
    if missing_settings:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_settings)}")
    
    return True
