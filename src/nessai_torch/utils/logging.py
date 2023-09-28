import logging


def configure_logger(log_level="INFO"):
    """Configure the logger"""
    logger = logging.getLogger("nessai_torch")
    logger.setLevel(log_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    return logger
