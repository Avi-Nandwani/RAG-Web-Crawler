"""
Configuration loader for RAG Web Crawler
Loads and validates configuration from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._override_from_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _override_from_env(self):
        """Override config values from environment variables"""
        # API settings
        if os.getenv("API_HOST"):
            self._config["api"]["host"] = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self._config["api"]["port"] = int(os.getenv("API_PORT"))
        
        # LLM settings
        if os.getenv("OLLAMA_BASE_URL"):
            self._config["llm"]["base_url"] = os.getenv("OLLAMA_BASE_URL")
        if os.getenv("OLLAMA_MODEL"):
            self._config["llm"]["model_name"] = os.getenv("OLLAMA_MODEL")
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            self._config["logging"]["level"] = os.getenv("LOG_LEVEL")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Args:
            key_path: Dot-separated path (e.g., 'crawler.max_pages')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Examples:
            >>> config.get('crawler.max_pages')
            30
            >>> config.get('api.port')
            8000
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def crawler(self) -> Dict[str, Any]:
        """Get crawler configuration"""
        return self._config.get("crawler", {})
    
    @property
    def chunking(self) -> Dict[str, Any]:
        """Get chunking configuration"""
        return self._config.get("chunking", {})
    
    @property
    def embeddings(self) -> Dict[str, Any]:
        """Get embeddings configuration"""
        return self._config.get("embeddings", {})
    
    @property
    def vectorstore(self) -> Dict[str, Any]:
        """Get vector store configuration"""
        return self._config.get("vectorstore", {})
    
    @property
    def retrieval(self) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return self._config.get("retrieval", {})
    
    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self._config.get("llm", {})
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self._config.get("api", {})
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self._config.get("paths", {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self._config.get("logging", {})


# Global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration loading
    print("Configuration loaded successfully!")
    print(f"Max pages: {config.get('crawler.max_pages')}")
    print(f"Chunk size: {config.get('chunking.chunk_size')}")
    print(f"API port: {config.get('api.port')}")
    print(f"LLM model: {config.get('llm.model_name')}")
