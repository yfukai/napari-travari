import logging
import os
from os import path

def get_logger(subpath=".tracking_log/log.txt"):
    log_path=path.join(path.expanduser("~"),subpath)
    if not path.exists(path.dirname(log_path)):
        os.makedirs(path.dirname(log_path))
    logging.basicConfig(filename=log_path,
                        level=logging.INFO)
    logger=logging.getLogger(__name__)
    return logger

def log_error(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args,**kwargs)
        except BaseException as error:
            logger.error(traceback.format_exc())
    return wrapped
 