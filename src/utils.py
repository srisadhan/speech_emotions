from contextlib import contextmanager
import numpy as np

_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 


class skip(object):
    """A decorator to skip function execution.

    Parameters
    ----------
    f : function
        Any function whose execution need to be skipped.

    Attributes
    ----------
    f

    """
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        print('skipping : ' + self.f.__name__)


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """
    @contextmanager
    def check_active():
        deactivated = ['skip']
        if flag in deactivated:
            print('Skipping the block: ' + f)
            raise SkipWith()
        else:
            print('Running the block: ' + f)
            yield

    try:
        yield check_active
    except SkipWith:
        pass
