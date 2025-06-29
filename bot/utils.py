"""
Utility functions for the Telegram bot
"""
import re
from typing import List, Optional, Any
from cryptography.fernet import Fernet
from config.settings import settings
import base64
import structlog

logger = structlog.get_logger(__name__)

def format_currency(amount: float, currency: str = "IDR") -> str:
    """Format currency with proper thousand separators"""
    if currency.upper() == "IDR":
        return f"Rp {amount:,.0f}".replace(',', '.')
    else:
        if amount >= 1:
            return f"{amount:,.2f}"
        else:
            return f"{amount:.8f}"

def format_percentage(value: float) -> str:
    """Format percentage with + or - sign"""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"

def is_admin_user(user_id: int) -> bool:
    """Check if user is admin"""
    return user_id in settings.telegram_admin_ids

def validate_pair_id(pair_id: str) -> bool:
    """Validate trading pair ID format"""
    pattern = r'^[a-z]+_idr$'
    return bool(re.match(pattern, pair_id.lower()))

def validate_amount(amount_str: str) -> Optional[float]:
    """Validate and parse amount string"""
    try:
        # Remove common separators and spaces
        cleaned = re.sub(r'[,.\s]', '', amount_str)
        amount = float(cleaned)
        
        if amount <= 0:
            return None
        
        return amount
    except (ValueError, TypeError):
        return None

def round_coin_amount(amount: float) -> float:
    """Round coin amount to 8 decimal places for Indodax API compliance"""
    return round(amount, 8)

def validate_coin_amount(amount: float) -> bool:
    """Validate if coin amount has valid precision (8 decimal places or less)"""
    # Convert to string and check decimal places
    amount_str = str(amount)
    if '.' in amount_str:
        decimal_places = len(amount_str.split('.')[1])
        return decimal_places <= 8
    return True

def validate_api_key(api_key: str) -> bool:
    """Basic validation for API key format"""
    if not api_key or len(api_key) < 20:
        return False
    
    # Check if it contains only valid characters
    pattern = r'^[A-Za-z0-9\-_]+$'
    return bool(re.match(pattern, api_key))

def get_crypto_symbol(pair_id: str) -> str:
    """Extract crypto symbol from pair ID"""
    return pair_id.split('_')[0].upper()

def get_crypto_emoji(symbol: str) -> str:
    """Get emoji for cryptocurrency"""
    emoji_map = {
        'BTC': '₿',
        'ETH': '⟠',
        'BNB': '🔸',
        'ADA': '🔷',
        'SOL': '☀️',
        'DOT': '🔴',
        'LINK': '🔗',
        'UNI': '🦄',
        'LTC': '🥈',
        'XRP': '💧',
        'MATIC': '🟣',
        'AVAX': '🏔️'
    }
    return emoji_map.get(symbol.upper(), '💰')

def truncate_text(text: str, max_length: int = 4000) -> str:
    """Truncate text to fit Telegram message limits"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def parse_command_args(text: str) -> List[str]:
    """Parse command arguments from message text"""
    parts = text.split()
    return parts[1:] if len(parts) > 1 else []

def format_timeframe(timeframe: str) -> str:
    """Format timeframe for display"""
    timeframe_map = {
        '1': '1 menit',
        '15': '15 menit', 
        '60': '1 jam',
        '240': '4 jam',
        '1D': '1 hari',
        '1W': '1 minggu'
    }
    return timeframe_map.get(timeframe, timeframe)

def calculate_profit_loss(buy_price: float, current_price: float, amount: float) -> dict:
    """Calculate profit/loss for a position"""
    if buy_price <= 0:
        return {"pnl": 0, "percentage": 0}
    
    pnl = (current_price - buy_price) * amount
    percentage = ((current_price - buy_price) / buy_price) * 100
    
    return {
        "pnl": pnl,
        "percentage": percentage,
        "is_profit": pnl >= 0
    }

def get_risk_level_emoji(risk_level: str) -> str:
    """Get emoji for risk level"""
    risk_emojis = {
        'low': '🟢',
        'medium': '🟡', 
        'high': '🔴'
    }
    return risk_emojis.get(risk_level.lower(), '⚪')

def format_order_status(status: str, language: str = "id") -> str:
    """Format order status with emoji and translation"""
    status_map = {
        "id": {
            "pending": "🟡 Menunggu",
            "completed": "✅ Selesai",
            "cancelled": "❌ Dibatalkan",
            "partial": "🟠 Sebagian"
        },
        "en": {
            "pending": "🟡 Pending",
            "completed": "✅ Completed", 
            "cancelled": "❌ Cancelled",
            "partial": "🟠 Partial"
        }
    }
    
    return status_map.get(language, status_map["id"]).get(status.lower(), status)

def clean_pair_id(pair_input: str) -> str:
    """Clean and standardize pair ID"""
    # Remove common separators and convert to lowercase
    cleaned = re.sub(r'[/\-_\s]', '_', pair_input.lower())
    
    # Ensure it ends with _idr
    if not cleaned.endswith('_idr'):
        if cleaned.endswith('idr'):
            cleaned = cleaned[:-3] + '_idr'
        else:
            cleaned += '_idr'
    
    return cleaned

def mask_api_key(api_key: str) -> str:
    """Mask API key for display"""
    if not api_key or len(api_key) < 8:
        return "****"
    
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]

# Encryption utilities
def get_encryption_key() -> bytes:
    """Get or generate encryption key"""
    # Use secret key from settings as basis for encryption key
    key_material = settings.secret_key.encode()
    
    # Pad or truncate to 32 bytes
    if len(key_material) < 32:
        key_material = key_material + b'0' * (32 - len(key_material))
    else:
        key_material = key_material[:32]
    
    # Encode to base64 URL-safe format for Fernet
    return base64.urlsafe_b64encode(key_material)

def encrypt_api_key(api_key: str) -> str:
    """Encrypt API key for storage"""
    try:
        fernet = Fernet(get_encryption_key())
        encrypted = fernet.encrypt(api_key.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    except Exception as e:
        logger.error("Failed to encrypt API key", error=str(e))
        raise

def decrypt_api_key(encrypted_api_key: str) -> str:
    """Decrypt API key from storage"""
    try:
        fernet = Fernet(get_encryption_key())
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_api_key.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        logger.error("Failed to decrypt API key", error=str(e))
        raise

def validate_trade_amount(amount: float, balance: float, min_amount: float = 10000) -> dict:
    """Validate trade amount against balance and limits"""
    result = {
        "valid": True,
        "error": None,
        "warning": None
    }
    
    if amount <= 0:
        result["valid"] = False
        result["error"] = "Jumlah harus lebih dari 0"
        return result
    
    if amount < min_amount:
        result["valid"] = False
        result["error"] = f"Jumlah minimum adalah {format_currency(min_amount)}"
        return result
    
    if amount > balance:
        result["valid"] = False
        result["error"] = "Saldo tidak mencukupi"
        return result
    
    # Warning if amount is more than 50% of balance
    if amount > balance * 0.5:
        result["warning"] = "Anda akan menggunakan lebih dari 50% saldo"
    
    return result

def get_market_status() -> dict:
    """Get current market status (simplified)"""
    # This would typically check exchange status, trading hours, etc.
    return {
        "is_open": True,
        "status": "normal",
        "message": None
    }

def format_volume(volume: float) -> str:
    """Format trading volume for display"""
    if volume >= 1_000_000_000:
        return f"{volume/1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"{volume/1_000_000:.2f}M" 
    elif volume >= 1_000:
        return f"{volume/1_000:.2f}K"
    else:
        return f"{volume:.2f}"

def parse_timeframe_input(timeframe_input: str) -> Optional[str]:
    """Parse user timeframe input to standard format"""
    timeframe_map = {
        '1m': '1',
        '1 menit': '1',
        '15m': '15', 
        '15 menit': '15',
        '1h': '60',
        '1 jam': '60',
        '4h': '240',
        '4 jam': '240', 
        '1d': '1D',
        '1 hari': '1D',
        '1w': '1W',
        '1 minggu': '1W'
    }
    
    normalized = timeframe_input.lower().strip()
    return timeframe_map.get(normalized)

def escape_markdown(text: str) -> str:
    """Escape markdown characters for Telegram"""
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    
    return text

def get_signal_strength_emoji(confidence: float) -> str:
    """Get emoji based on signal confidence"""
    if confidence >= 0.8:
        return "🔥"  # Very strong
    elif confidence >= 0.6:
        return "💪"  # Strong
    elif confidence >= 0.4:
        return "👍"  # Moderate
    else:
        return "🤏"  # Weak
