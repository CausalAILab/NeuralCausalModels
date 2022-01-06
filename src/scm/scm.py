import itertools

import torch as T
import torch.nn as nn

from .distribution import Distribution


class SCM(nn.Module):
    def __init__(self, v, f, pu: Distribution):
        super().__init__()
        self.v = v
        self.u = list(pu)
        self.f = f
        self.pu = pu
        self.device_param = nn.Parameter(T.empty(0))

    def space(self, select=None, tensor=True):
        if select is None:
            select = self.v
        for pairs in itertools.product(*([
                (vi, T.LongTensor([[0]]).to(self.device_param.device) if tensor else 0),
                (vi, T.LongTensor([[1]]).to(self.device_param.device) if tensor else 1)]
                                         for vi in select)):
            yield dict(pairs)

    def forward(self, n=None, u=None, do={}, select=None):
        assert not set(do.keys()).difference(self.v)
        assert (n is None) != (u is None)
        if u is None:
            u = self.pu.sample(n)
        if select is None:
            select = self.v
        v = {}
        remaining = set(select)
        for k in self.v:
            v[k] = do[k] if k in do else self.f[k](v, u)
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}
