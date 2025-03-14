import logging
import sys

def configure_global_logger(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )

    logging.info("Global logger configured at level: %s", logging.getLevelName(level))