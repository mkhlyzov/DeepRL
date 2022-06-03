import functools
import time

from drl.utils.common import GET_TIME


def highlight_output(func):
    @functools.wraps(func)
    def decorated(*args, **kwargs):
        print('============================================================')
        print(f'Starting {func.__name__!r}')
        t0 = GET_TIME()
        ret = func(*args, **kwargs)
        run_time = GET_TIME() - t0
        print(f'Finished {func.__name__!r} in {run_time:.4f} seconds')
        print('============================================================')
        return ret
    return decorated


def slow_down(_func=None, *, seconds=1):
    """Sleep for <seconds> before calling a function"""
    def slow_down_decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            time.sleep(seconds)
            return func(*args, **kwargs)
        return decorated

    if _func is None:
        return slow_down_decorator
    else:
        return slow_down_decorator(_func)
