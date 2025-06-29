"""
Database migration and initialization script for Telebot AI Trading Bot
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
from config.settings import settings
from core.database import Base, get_db, User, Trade, Portfolio, AISignal, UserSettings
import structlog

logger = structlog.get_logger(__name__)

class DatabaseMigrator:
    """Database migration and initialization manager"""
    
    def __init__(self):
        self.sync_engine = create_engine(
            settings.database_url.replace('postgresql+asyncpg', 'postgresql'),
            pool_pre_ping=True
        )
        self.async_engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True
        )
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Extract database name from URL
            db_name = settings.database_url.split('/')[-1].split('?')[0]
            
            # Connect to postgres database to create our database
            base_url = settings.database_url.rsplit('/', 1)[0] + '/postgres'
            base_engine = create_engine(
                base_url.replace('postgresql+asyncpg', 'postgresql'),
                isolation_level='AUTOCOMMIT'
            )
            
            with base_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": db_name}
                )
                
                if not result.fetchone():
                    logger.info(f"Creating database: {db_name}")
                    conn.execute(text(f"CREATE DATABASE {db_name}"))
                    logger.info("Database created successfully")
                else:
                    logger.info("Database already exists")
            
            base_engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    def create_tables(self):
        """Create all database tables"""
        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(self.sync_engine)
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            logger.warning("Dropping all database tables...")
            Base.metadata.drop_all(self.sync_engine)
            logger.info("Database tables dropped successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            return False
    
    async def test_connection(self):
        """Test database connection"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed")
                    return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def create_indexes(self):
        """Create database indexes for better performance"""
        try:
            logger.info("Creating database indexes...")
            
            with self.sync_engine.connect() as conn:
                # Users table indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_users_telegram_id 
                    ON users(telegram_id)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_users_is_active 
                    ON users(is_active)
                """))
                
                # Trades table indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_trades_user_id 
                    ON trades(user_id)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_trades_symbol 
                    ON trades(symbol)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_trades_created_at 
                    ON trades(created_at)
                """))
                
                # Portfolio table indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_portfolio_user_id 
                    ON portfolio(user_id)
                """))
                
                # AI Signals table indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_aisignals_symbol 
                    ON aisignals(symbol)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_aisignals_created_at 
                    ON aisignals(created_at)
                """))
                
                conn.commit()
                logger.info("Database indexes created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False
    
    def seed_initial_data(self):
        """Seed initial data into the database"""
        try:
            logger.info("Seeding initial data...")
            
            with self.sync_engine.connect() as conn:
                # Check if we already have data
                result = conn.execute(text("SELECT COUNT(*) FROM users"))
                user_count = result.fetchone()[0]
                
                if user_count > 0:
                    logger.info("Database already has data, skipping seed")
                    return True
                
                # Add any initial data here
                # For example, default settings, admin users, etc.
                
                conn.commit()
                logger.info("Initial data seeded successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to seed initial data: {e}")
            return False
    
    def backup_database(self, backup_path: str = None):
        """Create a database backup"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"data/backups/db_backup_{timestamp}.sql"
            
            logger.info(f"Creating database backup: {backup_path}")
            
            # Create backup directory if it doesn't exist
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Extract database connection details
            import urllib.parse
            parsed_url = urllib.parse.urlparse(settings.database_url)
            
            host = parsed_url.hostname
            port = parsed_url.port or 5432
            username = parsed_url.username
            password = parsed_url.password
            database = parsed_url.path.lstrip('/')
            
            # Use pg_dump to create backup
            import subprocess
            import os
            
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            cmd = [
                'pg_dump',
                '-h', host,
                '-p', str(port),
                '-U', username,
                '-d', database,
                '-f', backup_path,
                '--no-password'
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Database backup created successfully")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            return False
    
    def restore_database(self, backup_path: str):
        """Restore database from backup"""
        try:
            logger.info(f"Restoring database from: {backup_path}")
            
            if not Path(backup_path).exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Extract database connection details
            import urllib.parse
            parsed_url = urllib.parse.urlparse(settings.database_url)
            
            host = parsed_url.hostname
            port = parsed_url.port or 5432
            username = parsed_url.username
            password = parsed_url.password
            database = parsed_url.path.lstrip('/')
            
            # Use psql to restore backup
            import subprocess
            import os
            
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            cmd = [
                'psql',
                '-h', host,
                '-p', str(port),
                '-U', username,
                '-d', database,
                '-f', backup_path,
                '--no-password'
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Database restored successfully")
                return True
            else:
                logger.error(f"Database restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            return False
    
    def get_migration_status(self):
        """Get current migration status"""
        try:
            with self.sync_engine.connect() as conn:
                # Check if tables exist
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                
                tables = [row[0] for row in result.fetchall()]
                
                # Check for expected tables
                expected_tables = ['users', 'trades', 'portfolio', 'aisignals', 'usersettings']
                missing_tables = [t for t in expected_tables if t not in tables]
                
                return {
                    'existing_tables': tables,
                    'missing_tables': missing_tables,
                    'migration_complete': len(missing_tables) == 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return None

async def main():
    """Main migration function"""
    migrator = DatabaseMigrator()
    
    print("🗄️  Telebot AI Trading Bot - Database Migration")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python migrate_db.py <command>")
        print("Commands:")
        print("  init     - Initialize database (create DB, tables, indexes)")
        print("  create   - Create tables only")
        print("  drop     - Drop all tables (DANGER!)")
        print("  test     - Test database connection")
        print("  status   - Show migration status")
        print("  backup   - Create database backup")
        print("  restore <file> - Restore from backup")
        print("  seed     - Seed initial data")
        return
    
    command = sys.argv[1].lower()
    
    if command == "init":
        print("🔧 Initializing database...")
        
        if migrator.create_database_if_not_exists():
            print("✅ Database created/verified")
        else:
            print("❌ Failed to create database")
            return
        
        if migrator.create_tables():
            print("✅ Tables created")
        else:
            print("❌ Failed to create tables")
            return
        
        if migrator.create_indexes():
            print("✅ Indexes created")
        else:
            print("❌ Failed to create indexes")
            return
        
        if migrator.seed_initial_data():
            print("✅ Initial data seeded")
        else:
            print("❌ Failed to seed initial data")
            return
        
        print("🎉 Database initialization complete!")
    
    elif command == "create":
        if migrator.create_tables():
            print("✅ Tables created successfully")
        else:
            print("❌ Failed to create tables")
    
    elif command == "drop":
        confirm = input("⚠️  This will delete ALL data! Type 'DELETE' to confirm: ")
        if confirm == "DELETE":
            if migrator.drop_tables():
                print("✅ Tables dropped successfully")
            else:
                print("❌ Failed to drop tables")
        else:
            print("Operation cancelled")
    
    elif command == "test":
        print("🔌 Testing database connection...")
        if await migrator.test_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
    
    elif command == "status":
        print("📊 Checking migration status...")
        status = migrator.get_migration_status()
        if status:
            print(f"Existing tables: {', '.join(status['existing_tables'])}")
            if status['missing_tables']:
                print(f"Missing tables: {', '.join(status['missing_tables'])}")
            print(f"Migration complete: {'✅ Yes' if status['migration_complete'] else '❌ No'}")
    
    elif command == "backup":
        if migrator.backup_database():
            print("✅ Database backup created")
        else:
            print("❌ Failed to create backup")
    
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ Please specify backup file path")
            return
        
        backup_file = sys.argv[2]
        confirm = input(f"⚠️  This will restore from {backup_file}! Type 'RESTORE' to confirm: ")
        if confirm == "RESTORE":
            if migrator.restore_database(backup_file):
                print("✅ Database restored successfully")
            else:
                print("❌ Failed to restore database")
        else:
            print("Operation cancelled")
    
    elif command == "seed":
        if migrator.seed_initial_data():
            print("✅ Initial data seeded")
        else:
            print("❌ Failed to seed initial data")
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    asyncio.run(main())
