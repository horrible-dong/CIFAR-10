import os
import threading

from torch import distributed as dist
from tqdm import tqdm


def update(model, loss, optimizer):
    model.zero_grad()
    loss.backward()
    optimizer.step()


def reduce_value(value, average=True):
    world_size = float(dist.get_world_size())
    dist.all_reduce(value)
    if average:
        value /= world_size
    return value


def async_exec(target, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    threading.Thread(target=target, args=args, kwargs=kwargs).start()


def mkdir(path, mode=0o777):
    if not os.path.exists(path):
        os.mkdir(path)
        os.chmod(path, mode)


def makedirs(path, mode=0o777, exist_ok=False):
    head, tail = os.path.split(path)
    if not tail:
        head, tail = os.path.split(head)
    if head and tail and not os.path.exists(head):
        try:
            makedirs(head, exist_ok=exist_ok)
        except FileExistsError:
            pass
        cdir = os.path.curdir
        if isinstance(tail, bytes):
            cdir = bytes(os.path.curdir, 'ASCII')
        if tail == cdir:
            return
    try:
        mkdir(path, mode)
    except OSError:
        if not exist_ok or not os.path.isdir(path):
            raise
