"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Configuration
    LLM_MODEL = "gemini-2.5-flash"
    TEMPERATURE = 0
    MAX_TOKENS = None
    TIMEOUT = None
    MAX_RETRIES = 2
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/"
    ]
    
    @classmethod
    def validate_api_key(cls):
        """Validate and ensure GOOGLE_API_KEY is available"""
        if not cls.GOOGLE_API_KEY:
            cls.GOOGLE_API_KEY = getpass.getpass("Enter your Google AI API key: ")
        os.environ["GOOGLE_API_KEY"] = cls.GOOGLE_API_KEY
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        cls.validate_api_key()
        
        return ChatGoogleGenerativeAI(
            model=cls.LLM_MODEL,
            temperature=cls.TEMPERATURE,
            max_tokens=cls.MAX_TOKENS,
            timeout=cls.TIMEOUT,
            max_retries=cls.MAX_RETRIES
        )