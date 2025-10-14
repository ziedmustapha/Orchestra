import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Model Configuration
    llama3_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct", 
        env="LLAMA3_MODEL"
    )
    gemma3_model: str = Field(
        default="google/gemma-2-27b-it",  # Using Gemma 2 as example
        env="GEMMA3_MODEL"
    )
    
    # vLLM Configuration
    tensor_parallel_size: int = Field(default=2, env="TENSOR_PARALLEL_SIZE")
    max_num_seqs: int = Field(default=256, env="MAX_NUM_SEQS")
    gpu_memory_utilization: float = Field(default=0.9, env="GPU_MEMORY_UTILIZATION")
    enable_prefix_caching: bool = Field(default=True, env="ENABLE_PREFIX_CACHING")
    
    # API Configuration
    max_model_len: int = Field(default=4096, env="MAX_MODEL_LEN")
    default_max_tokens: int = Field(default=256, env="DEFAULT_MAX_TOKENS")
    
    class Config:
        env_file = ".env"
        
settings = Settings()