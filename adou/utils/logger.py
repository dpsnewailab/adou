import time
import logging
import sys


def get_logger(logger_name='default', *args, **kwargs):
    """
    Get logging and format
    All logs will be saved into logs/log-DATE (default)
    Default size of log file = 15m
    :param logger_name:
    :return:
    """
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    log.addHandler(ch)

    return log


def timer(func, logger=None, *args, **kwargs):
    """
    Timer decorator
    """
    if logger is None:
        logger = get_logger(logger_name='Timer')

    def wrapper(*args, **kwargs):
        before = time.time()
        rv = func(*args, **kwargs)
        after = time.time()
        logger.info(f'{func.__name__} took {round(after - before, 5)}s for execution.')
        return rv
    
    return wrapper


def ignore_runtime_error(func, logger=None, *args, **kwargs):
    """
    Conveniently try..catch
    """
    if logger is None:
        logger = get_logger(logger_name='Error')

    def wrapper(*args, **kwargs):
        try:
            rv = func(*args, **kwargs)
        except Exception as error:
            logger.info(f'[{func.__name__}] {error}')
            rv = None
        finally:
            return rv
    
    return wrapper