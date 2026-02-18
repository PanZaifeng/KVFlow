"""
Create logger named agentsociety as singleton for ray.
"""
import os

import logging
from datetime import datetime

PF_LEVEL = 25  # between INFO 20 and WARNING 30
logging.addLevelName(PF_LEVEL, "SD")

def pf(self, message, *args, **kwargs):
    if self.isEnabledFor(PF_LEVEL):
        self._log(PF_LEVEL, message, args, **kwargs)

logging.Logger.PF = pf

class PFLevelFilter(logging.Filter):
    """过滤器，只对名称为'PFEngine'的logger允许 PF 级别的日志通过，其他logger不受限制"""
    def filter(self, record):
        # 如果logger名称是PFEngine，只允许PF级别通过
        if record.name == "PFEngine":
            return record.levelno == PF_LEVEL
        # 其他logger不受此过滤器影响，都允许通过
        return True

__all__ = ["get_logger", "set_logger_level", "PF_LEVEL"]


def _setup_logger():
    logger = logging.getLogger("PFEngine")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        log_dir = os.path.join(os.getcwd(), "log")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"pf_{timestamp}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

_logger = _setup_logger()

def get_logger():
    return _logger


def set_logger_level(level: str, logger: logging.Logger):
    """Set the logger level"""
    
    if level.upper() == "PF":
        logger.setLevel(PF_LEVEL)
        for handler in logger.handlers:
            handler.filters.clear()
            handler.addFilter(PFLevelFilter())
    else:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.filters.clear()
