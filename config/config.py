import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Model Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models")
    
    # File Upload Configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = [".pdf", ".docx"]
    TEMP_DIR = "temp"
    
    # Matching Algorithm Configuration
    MATCHING_WEIGHTS = {
        "skills_match": 0.5,
        "experience_match": 0.3,
        "education_match": 0.2
    }
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8501"))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    @staticmethod
    def validate():
        """Validate configuration"""
        # Ensure required directories exist
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        
        # Validate file size limit
        if Config.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
            
        # Validate matching weights
        total_weight = sum(Config.MATCHING_WEIGHTS.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Matching weights must sum to 1.0") 