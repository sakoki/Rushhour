import functools
import time

def timer(func):
    '''Prints the runtime of decorated function'''

    @functools.wraps(func)
    def wrap_timer(*args, **kwargs):
        print('Running {}...'.format(func.__name__))
        start_time = time.perf_counter()
        results = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print('Finished in {:.4f}s'.format(run_time))
        return results
    return wrap_timer