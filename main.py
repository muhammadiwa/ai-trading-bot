"""
Main entry point for the Telegram Trading Bot AI
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings, validate_settings
from bot.telegram_bot import TelegramBot
from core.database import init_database
from core.scheduler import init_scheduler
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

async def main():
    """Main application entry point"""
    try:
        # Validate configuration
        logger.info("Starting Telegram Trading Bot AI...")
        validate_settings()
        logger.info("Configuration validated successfully")
        
        # Initialize database
        logger.info("Initializing database...")
        await init_database()
        logger.info("Database initialized successfully")
        
        # Initialize scheduler for automated tasks
        logger.info("Initializing scheduler...")
        scheduler = init_scheduler()
        scheduler.start()
        logger.info("Scheduler started successfully")
        
        # Initialize and start Telegram bot
        logger.info("Starting Telegram bot...")
        bot = TelegramBot()
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error("Failed to start application", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down application...")
        # Cleanup code here
        sys.exit(0)

if __name__ == "__main__":
    # Set up basic logging for startup
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        sys.exit(1)
