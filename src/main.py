"""
Main entry point for Omni-Assistant Discord Bot.
Initializes all components and starts the bot.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from discord_handlers import create_discord_bot
from llm_integration import llm_manager
from memory_manager import memory_manager

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    log_file = Path(config.log_file)
    log_file.parent.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(config.log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

async def startup_checks():
    """Perform startup checks and initialization."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Omni-Assistant Discord Bot...")
    logger.info(f"Configuration loaded: LM Studio at {config.lm_studio_url}")
    
    # Check LM Studio connection
    logger.info("Checking LM Studio connection...")
    try:
        health = await llm_manager.client.health_check()
        if health:
            logger.info("[OK] LM Studio connection successful")
        else:
            logger.warning("[WARNING] LM Studio health check failed - bot will still start but may have limited functionality")
    except Exception as e:
        logger.error(f"[ERROR] LM Studio connection error: {e}")
        logger.warning("Bot will start but LLM functionality may be limited")

    # Check vector database
    logger.info("Checking vector database...")
    if memory_manager.chroma_client:
        logger.info("[OK] Vector database initialized successfully")
    else:
        logger.warning("[WARNING] Vector database initialization failed - memory features may be limited")
    
    # Check Discord configuration
    if not config.discord_bot_token:
        logger.error("[ERROR] Discord bot token not found in environment variables")
        return False

    if config.discord_channel_ids:
        logger.info(f"Bot will monitor specific channels: {config.discord_channel_ids}")
    else:
        logger.info("Bot will monitor all channels (no channel restriction)")

    logger.info("[OK] Startup checks completed")
    return True

async def shutdown_handler(bot):
    """Handle graceful shutdown."""
    logger = logging.getLogger(__name__)
    logger.info("Shutting down Omni-Assistant...")
    
    try:
        # Close Discord connection
        if not bot.is_closed():
            await bot.close()
        
        # Close LLM manager
        await llm_manager.close()

        logger.info("[OK] Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def signal_handler(signum, frame, bot):
    """Handle system signals for graceful shutdown."""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, initiating shutdown...")
    
    # Create new event loop for shutdown if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(shutdown_handler(bot))
    sys.exit(0)

async def main():
    """Main function to run the bot."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Perform startup checks
        startup_success = await startup_checks()
        if not startup_success:
            logger.error("Startup checks failed. Exiting.")
            return
        
        # Create and configure bot
        bot = create_discord_bot()
        
        # Set up signal handlers for graceful shutdown
        def make_signal_handler(bot):
            def handler(signum, frame):
                signal_handler(signum, frame, bot)
            return handler
        
        signal.signal(signal.SIGINT, make_signal_handler(bot))
        signal.signal(signal.SIGTERM, make_signal_handler(bot))
        
        # Start the bot
        logger.info("[STARTING] Discord bot...")
        await bot.start(config.discord_bot_token)
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Ensure cleanup
        if 'bot' in locals():
            await shutdown_handler(bot)

def run_bot():
    """Entry point for running the bot."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Print startup banner
    print("""
    ==========================================
               Omni-Assistant
            Discord AI Bot v1.0

      Powered by CrewAI & LM Studio
      Local AI - Multi-Agent - Memory
    ==========================================
    """)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run the bot
    run_bot()
