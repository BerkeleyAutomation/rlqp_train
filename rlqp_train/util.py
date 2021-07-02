import torch
import torch.nn as nn

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
