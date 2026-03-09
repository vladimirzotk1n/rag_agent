from loguru import logger

logger.add(
    "prod.log", level="INFO", rotation="10 MB", retention="1 month", compression="gz"
)
