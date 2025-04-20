from time import perf_counter
from contextlib import ContextDecorator

""" Answer 7 by Greg in https://stackoverflow.com/questions/33987060 """
""" (python-context-manager-that-measures-time)                      """
""" The timer is a context manager, use with "with" statement        """    

class Timer(ContextDecorator):
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        elapsed = perf_counter() - self.time
        print(f'{self.msg}: {elapsed:.2f}s')
        