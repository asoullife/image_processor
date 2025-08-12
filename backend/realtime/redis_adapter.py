"""Redis adapter for Socket.IO multi-process support."""

import socketio
import logging
from typing import Optional
import os

# Handle aioredis import with fallback
try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger(__name__)

class RedisAdapter:
    """Redis adapter for Socket.IO scaling across multiple processes."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client: Optional[aioredis.Redis] = None
        self.adapter: Optional[socketio.AsyncRedisManager] = None

    async def initialize(self) -> Optional[socketio.AsyncRedisManager]:
        """Initialize Redis adapter for Socket.IO."""
        if not REDIS_AVAILABLE:
            logger.warning("aioredis not available, Socket.IO will run in single-process mode")
            return None
            
        try:
            # Create Redis client
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            # Test Redis connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")

            # Create Socket.IO Redis adapter
            self.adapter = socketio.AsyncRedisManager(
                self.redis_url,
                write_only=False,
                logger=logger
            )

            logger.info("Redis adapter initialized successfully")
            return self.adapter

        except Exception as e:
            logger.warning(f"Failed to initialize Redis adapter: {e}")
            logger.info("Socket.IO will run in single-process mode")
            return None

    async def cleanup(self):
        """Clean up Redis connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis client closed")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")

    async def cache_progress(self, session_id: str, progress_data: dict, ttl: int = 3600):
        """Cache progress data in Redis."""
        if not REDIS_AVAILABLE:
            return
            
        try:
            if self.redis_client:
                key = f"progress:{session_id}"
                await self.redis_client.setex(key, ttl, str(progress_data))
                logger.debug(f"Cached progress for session {session_id}")
        except Exception as e:
            logger.error(f"Error caching progress: {e}")

    async def get_cached_progress(self, session_id: str) -> Optional[dict]:
        """Get cached progress data from Redis."""
        if not REDIS_AVAILABLE:
            return None
            
        try:
            if self.redis_client:
                key = f"progress:{session_id}"
                data = await self.redis_client.get(key)
                if data:
                    import json
                    return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting cached progress: {e}")
        return None

    async def clear_session_cache(self, session_id: str):
        """Clear cached data for a session."""
        if not REDIS_AVAILABLE:
            return
            
        try:
            if self.redis_client:
                key = f"progress:{session_id}"
                await self.redis_client.delete(key)
                logger.debug(f"Cleared cache for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing session cache: {e}")

# Global Redis adapter instance
redis_adapter = RedisAdapter()