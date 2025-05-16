import logging
import os
from datetime import datetime

log_dir="logs"

os.makedirs(log_dir, exist_ok=True)

log_file=os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=log_file,
    format='%(asctime)s - %(levelname)s -%(message)s',
    level=logging.INFO
)

def get_logger(name):
    logger= logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger