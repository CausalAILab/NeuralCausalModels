import itertools

import numpy as np
import torch as T
import torch.nn as nn

from src.ds import CausalGraph

from .distribution import Distribution


class DiscreteDistribution(Distribution):
    def log_pmf(self, u):
        raise NotImplementedError()

    def space(self):
        raise NotImplementedError()


class FactorizedDistribution(DiscreteDistribution):
    def simplex_init(s):
        t = T.rand(s).flatten()
        t[-1] = 1
        t = t.sort()[0]
        t[1:] = t[1:] - t[:-1]
        return T.log(t.reshape(s))

    def __init__(self, us, cond={}, init='simplex'):
        us = list(map(lambda s: (s,) if type(
            s) is str else tuple(sorted(s)), us))
        super().__init__(list(ui for u in us for ui in u))
        if not all(k in self.u for vals in cond.values() for k in vals):
            raise ValueError('cond contains variables not in u')

        # sort us in topological order
        u2us = {u: next(usi for usi in us if u in usi) for u in self.u}
        self.us = list(CausalGraph(us, directed_edges=list(set((u2us[p], c)
                                                               for c in cond
                                                               for p in cond[c]))))
        self.cond = {u: sorted(cond.get(u, [])) for u in self.us}
        self.init = {
            'uniform': lambda s: T.zeros(s),  # wrong
            'simplex': FactorizedDistribution.simplex_init,
        }.get(init, init)
        self.q = nn.ParameterDict({
            str(us): nn.Parameter(self.init(tuple(2 for ui in itertools.chain(
                self.cond[us], us))))
            for us in self.us})

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device
        qs = {u: (self.q[str(u)][None].to(device)
                  + -T.log(-T.log(T.rand((n,) + tuple(self.q[str(u)].shape),
                                         device=device))))
              for u in self.us}
        qs = {}
        for us in self.us:
            t = self.q[str(us)]

            # select conditional probability distribution
            if self.cond[us]:
                t = t[tuple(qs[u2].flatten() for u2 in self.cond[us])]
            else:
                t = t.expand((n,) + tuple(t.shape))

            # sample using Gumbel-max
            for i in range(10):  # in case there are two maximums in one row
                # Gumbel-max
                gm = t + -T.log(-T.log(T.rand(t.shape, device=device)))
                gm = ((gm == (gm.view(n, -1).max(dim=1).values
                              .reshape((n,) + (1,) * (len(gm.shape) - 1))))
                      .nonzero(as_tuple=False)[:, 1:])
                if len(gm) == n:
                    break
            else:
                raise ValueError(
                    f'something went wrong! gm has shape {gm.shape}')

            # split samples into variables
            qs.update({us[i]: gm[:, i:i+1] for i in range(len(us))})
        return qs

    def log_pmf(self, u):
        return sum(
            (nn.functional.log_softmax(
                (self.q[str(us)]
                 .view(*((2,) * len(self.cond[us])), -1)
                 [tuple(u[u2] for u2 in self.cond[us])]),
                dim=-1)
             .reshape(self.q[str(us)].shape[len(self.cond[us]):])
             [tuple(u[k] for k in us)])
            for us in self.us)

    def space(self):
        for pairs in itertools.product(*(
            [(u, T.tensor([[0]])), (u, T.tensor([[1]]))]
                for u in self.u)):
            yield dict(pairs)


class BernoulliDistribution(DiscreteDistribution):
    def __init__(self, u_names, sizes, p, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        self.p = p
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p, size=(n, self.sizes[U]))).long().to(device)

        return u_vals
