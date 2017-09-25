import time
import datetime
import traceback
from multiprocessing.managers import SyncManager
import torch
import torch.multiprocessing as mp
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply

def parallel_worker(in_queue, out_queue, event):
    try:
        while True:
            #event.wait()
            t0 = datetime.datetime.now()
            module, args, kwargs, device = in_queue.get()
            t1 = datetime.datetime.now()
            with torch.cuda.device(device):
                y = module(*args, **kwargs)
            t2 = datetime.datetime.now()
            out_queue.put(y)
            print("[parallel_worker]",
                  "get", str(t1 - t0)[5:10],
                  "module", str(t2 - t1)[5:10],
                  "put", str(datetime.datetime.now() - t2)[5:10])

            module = None
            args = None
            kwargs = None
            y = None
            event.clear()

    except Exception as e:
        traceback.print_exc()
        out_queue.put(y)

class DataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

        ctx = mp.get_context("spawn")
        n_devices = len(self.device_ids)
        self.in_queues = tuple(ctx.Queue() for _ in range(n_devices))
        self.out_queues = tuple(ctx.Queue() for _ in range(n_devices))
        self.events = tuple(ctx.Event() for _ in range(n_devices))
        self.processes = []
        for i in range(n_devices):
            process = ctx.Process(target=parallel_worker,
                                  args=(self.in_queues[i], self.out_queues[i],
                                        self.events[i]))
            process.start()
            self.processes.append(process)

    def forward(self, *inputs, **kwargs):
        t0 = datetime.datetime.now()
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        t1 = datetime.datetime.now()
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        t2 = datetime.datetime.now()
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        t3 = datetime.datetime.now()
        out = self.gather(outputs, self.output_device)
        t4 = datetime.datetime.now()
        print("[forward]",
              "scatter", str(t1 - t0)[5:10],
              "replicate", str(t2 - t1)[5:10],
              "parallel_apply", str(t3 - t2)[5:10],
              "gather", str(t4 - t3)[5:10],
              "total", str(datetime.datetime.now() - t0)[5:10])

        return out

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(self.processes, self.in_queues, self.out_queues,
                              self.events, replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    """Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
