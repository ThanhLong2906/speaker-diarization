from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MILVUS_HOST: str 
    MILVUS_PORT: str
    GLOBAL_VOICE_DIR: str