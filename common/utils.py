import time
from functools import wraps

def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        t = time.process_time()
        result = f(*args, **kw)
        elapsed_time = time.process_time() - t
        print('func:{} took: {} sec'.format(f.__name__, elapsed_time))
        return result
    return wrap