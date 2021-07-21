import torch
import torch.nn as nn

activation_name_map = {
    "ReLU" : nn.ReLU
}

def mlp(sizes, activation, input_transform=None, output_activation=nn.Identity):
    layers = [] if input_transform is None else [input_transform]
    for j in range(len(sizes)-2):
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation()]
    layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
    return nn.Sequential(*layers)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, shape)
    else:
        return (length, *shape)

def freeze(m, frozen):
    for p in m.parameters():
        p.requires_grad = not frozen

class _Frozen:
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        freeze(self.module, True)
        return self.module

    def __exit__(self, type, value, traceback):
        freeze(self.module, False)

def frozen(module):
    return _Frozen(module)


class NonFuture:
    """Helper class for NonPool that is an already-evaluted Future"""
    def __init__(self, obj):
        self.obj = obj
    def get(self):
        return self.obj
    
class NonPool:
    """Non-threaded replacement for a mulitprocessing pool.

    Due to how the underlying OpenMP implementation works, we can either
    have multithreaded tensor evaluation and no child processes, or
    single-threaded tensor evaluation and child procesess.  When opting
    for multithreaded tensor evaluation, we need to avoid forks--that's
    where this class helps.
    """
    def apply_async(self, fn, args, kwargs, done_fn=None, err_fn=None):
        try:
            r = fn(*args, **kwargs)
        except ex:
            if err_fn:
                err_fn(ex)
                return
        done_fn(r)
        return NonFuture(r)
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass    
    def close(self):
        pass
    def join(self):
        pass
    
