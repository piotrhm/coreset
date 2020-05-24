import time
import numpy as np
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

@timeit
def cost_function(points, labels, centers):
    assignment = np.array([centers[labels[i]] for i in labels])
    cost = np.power(points-assignment, 2).sum(axis=1).sum()
    return cost

@timeit
def cost_function_2(points, labels, centers):
    assignment = np.array([map(lambda x: centers[x], labels)])
    cost = np.power(points-assignment, 2).sum(axis=1).sum()
    return cost