import time
import datetime
import traceback
import queue
import threading
import torch
import torch.multiprocessing
from torch.autograd import Variable


def get_a_var(obj):
    if isinstance(obj, Variable):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        results = map(get_a_var, obj)
        for result in results:
            if isinstance(result, Variable):
                return result
    if isinstance(obj, dict):
        results = map(get_a_var, obj.items())
        for result in results:
            if isinstance(result, Variable):
                return result
    return None

def parallel_apply(processes, pipes, modules, inputs,
                   kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    results = {}

    if len(modules) > 1:
        t0 = datetime.datetime.now()
        args = zip(pipes, modules, inputs, kwargs_tup, devices)
        [x[0][0].send(x[1:]) for x in args]
        t1 = datetime.datetime.now()
        t2 = datetime.datetime.now()
        results = {i: pipes[i][0].recv() for i in range(len(modules))}
        t3 = datetime.datetime.now()
        print("[parallel_apply]",
              "put", str(t1 - t0)[5:10],
              "set", str(t2 - t1)[5:10],
              "get", str(t3 - t2)[5:10],
              "total", str(datetime.datetime.now() - t0)[5:10])

    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], results, lock, devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
