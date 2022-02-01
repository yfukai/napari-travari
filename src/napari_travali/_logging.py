import logging
import traceback
from functools import wraps

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = True


def log_error(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as error:
            logger.error(traceback.format_exc())

    return wrapped
