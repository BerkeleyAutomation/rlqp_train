import torch
import torch.nn as nn

class Mode8(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 4

    def forward(self, v):
        # Ax, y, z, l, u, rho
        Ax = v[...,0]
        y = v[...,1]
        z = v[...,2]
        l = v[...,3]
        u = v[...,4]
        rho = v[...,5]

        lo = z - l
        hi = u - z

        return torch.stack([
            torch.log10(torch.clamp(torch.minimum(lo, hi), 1e-8, 1e6)),
            torch.clamp(y,      -1e6, 1e6),
            torch.clamp(z - Ax, -1e6, 1e6),
            torch.log10(torch.clamp(rho, 1e-6, 1e6))],
            axis=-1) #dim=1)
        
