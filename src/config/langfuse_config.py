"""Langfuse configuration and initialization."""

import os
from typing import Optional
from langfuse import Langfuse
from langfuse import observe


class LangfuseConfig:
    """Configuration class for Langfuse integration."""
    
    def __init__(self):
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.enabled = bool(self.secret_key and self.public_key)
        self._client: Optional[Langfuse] = None
    
    def get_client(self) -> Optional[Langfuse]:
        """Get Langfuse client instance."""
        if not self.enabled:
            return None
        
        if self._client is None:
            self._client = Langfuse(
                secret_key=self.secret_key,
                public_key=self.public_key,
                host=self.host
            )
        
        return self._client
    
    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled and properly configured."""
        return self.enabled


# Global Langfuse configuration instance
langfuse_config = LangfuseConfig()


def get_langfuse_client() -> Optional[Langfuse]:
    """Get the global Langfuse client instance."""
    return langfuse_config.get_client()


def is_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled."""
    return langfuse_config.is_enabled()


# Conditional decorator that only applies @observe if Langfuse is enabled
def conditional_observe(name: Optional[str] = None):
    """Decorator that applies @observe only if Langfuse is enabled."""
    def decorator(func):
        if is_langfuse_enabled():
            return observe(name=name)(func)
        return func
    return decorator