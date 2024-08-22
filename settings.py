from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mammo_unet_weights_path: str
    healthy_path: str
    diseased_path: str

    class Config:
        env_file = ".env"

settings = Settings()