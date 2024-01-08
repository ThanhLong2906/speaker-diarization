from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # SERVING_ENDPOINT: str
    MILVUS_HOST: str
    MILVUS_PORT: str
    # FACE_THRESHOLD: float