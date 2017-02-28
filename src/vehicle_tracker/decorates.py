# -*- coding: utf-8 -*-

import datetime
from functools import wraps


def log_spent_time(func):
    """
    A decorator used to log the time spent by the decorated function.
    """
    @wraps(func)
    def inner(*args):
        start = datetime.datetime.now()
        result = func(*args)
        end = datetime.datetime.now()
        spent_time = end - start
        print 'function {} spent {} microseconds'.format(
            func.__name__, spent_time.microseconds)
        return result

    return inner


def cache_func(func):
    """
    A decorator adds cache functionality to time consuming functions.
    """
    cache = {}

    @wraps(func)
    def inner(*args):
        args_hash = hash(str(args))
        if args_hash not in cache:
            cache[args_hash] = func(*args)

        return cache[args_hash]

    return inner

# EOF
