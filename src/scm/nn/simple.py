import torch as T
import torch.nn as nn

from . import MADE


class Simple(nn.Module):
    def __init__(self, v_size, u_size, o_size):
        super().__init__()
        self.v = sorted(v_size)
        self.u = sorted(u_size)
        self.v_size = v_size
        self.u_size = u_size
        self.o_size = o_size
        i = (sum(self.v_size[k] for k in self.v_size)
             + sum(self.u_size[k] for k in self.u_size)
             + o_size)
        h = max(128, i)
        self.nn = nn.Sequential(MADE(i, [h] * 2, i, natural_ordering=True),
                                nn.LogSigmoid())
        self.device_param = nn.Parameter(T.empty(0))

        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_normal_(m.weight,
                                     gain=T.nn.init.calculate_gain('relu'))

    def forward(self, pa, u, v=None, n=None):
        # confirm sampling / pmf estimation
        assert n is None or v is None, 'v and n may not both be set'
        estimation = v is not None

        # default number of samples to draw
        if n is None:
            n = 1

        # confirm sizes are correct
        for k in self.v_size:
            assert pa[k].shape[-1] == self.v_size[k], (
                k, pa[k].shape[-1], self.v_size[k])
        for k in self.u_size:
            assert u[k].shape[-1] == self.u_size[k], (
                k, u[k].shape[-1], self.u_size[k])

        if estimation:  # compute log P(v | pa_V, u_V)
            i = T.cat([pa[k] for k in self.v]
                      + [u[k] for k in self.u]
                      + [v], dim=-1)
            o = self.nn(i)
            o = o[..., -self.o_size:]
            o = T.where(v == 1, o, T.log(1 - 0.9999998 * T.exp(o) - 0.0000001))
            if (o >= 0).any():
                o[o >= 0] = -T.relu(-o[o >= 0]) - 0.0000001
            return o.sum(dim=-1)
        else:  # sample from P(V)
            if self.u:
                o_shape = tuple(u[next(k for k in self.u)].shape[:-1]) + (self.o_size,)
            else:
                o_shape = (n, self.o_size)

            if self.v or self.u:
                ib = T.cat([pa[k] for k in self.v]
                           + [u[k] for k in self.u], dim=-1)  # (n, dvu)
            else:
                ib = T.empty(n, 0).to(next(self.parameters()).device)

            # gumbel argmax per dimension
            o_acc = T.zeros(o_shape, device=self.device_param.device)  # (n, d)
            for d in range(self.o_size):
                i = T.cat([ib, o_acc], dim=-1)  # (n, dvu + d)
                o = self.nn(i)  # (n, dvu + d)
                o = o[..., ib.shape[-1] + d: ib.shape[-1] + d + 1]  # (n, 1)
                o = T.cat([T.log(1 - T.exp(o)), o], dim=-1)  # (n, 2)
                assert tuple(o.shape) == tuple(o_shape[:-1]) + (2,), (o.shape, o_shape)  # (n, d)
                g = -T.log(-T.log(T.rand(o.shape, device=self.device_param.device)))  # (n, 2)
                o_acc[..., d] = T.max(o + g, dim=-1).indices

            return o_acc


if __name__ == '__main__':
    s = Simple(dict(v1=2, v2=1), dict(u1=1, u2=2), 3)
    print(s)
    pa = {
        'v1': T.tensor([[1, 2], [3, 4.]]),
        'v2': T.tensor([[5], [6.]])
    }
    u = {
        'u1': T.tensor([[7.], [8]]),
        'u2': T.tensor([[9, 10], [11, 12.]])
    }
    v = T.tensor([[1, 2, 3], [4, 5, 6]]).float()
    print(s(pa, u, v))
    print(s(pa, u, n=1))

    import pandas as pd

    o = s(pa, u, n=10000)
    df = pd.DataFrame(o.numpy())
    print(df)
