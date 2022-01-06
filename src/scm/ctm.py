import itertools

import torch as T

from .distribution import FactorizedDistribution
from .scm import SCM

RPA = {
    'bidirected': dict(X=[], Y=[('X',)]),
    'backdoor': dict(Z=[], X=['Z'], Y=['X', 'Z']),
    'bow': dict(X=[], Y=['X', ('X',)]),
    'frontdoor': dict(X=[], M=['X'], Y=['M', ('X',)]),
    'iv': dict(I=[], X=['I'], Y=['X', ('X',)]),
    'm': dict(X=[], Y=['X'], M=[('X',), ('Y',)]),
    'napkin': dict(W=[('X',), ('Y',)], Z=['W'], X=['Z'], Y=['X']),
    'simple': dict(X=[], Y=['X']),
    'bad_m': dict(X=[], Z=[('X',), ('Y',)], Y=['X', 'Z']),
    'bad_m_2': dict(Z=[('X',), ('Y',)], X=['Z'], Y=['X']),
    'bdm': dict(Z=[('X',), ('Y',)], X=['Z'], Y=['X', 'Z']),
    'extended_bow': dict(X=[], Z=['X', ('X',)], Y=['Z']),
    'chain': dict(X=[], Z=['X'], Y=['Z']),
    'double_bow': dict(X=[], Z=['X', ('X',), ('Y',)], Y=['Z'])
}


class CTM(SCM):
    def __init__(self, cg_file=None, rpa=None, v_size={}):
        assert (cg_file is None) != (rpa is None)
        if cg_file is not None:
            name = cg_file.split('/')[-1].split('.')[0]
            if name not in RPA:
                raise ValueError(f"Graph '{name}' is unsupported")
            self.rpa = RPA[name]
        else:
            self.rpa = rpa
        self.r = {k: ((k, ()),) if not self.rpa[k] else tuple(sorted(
            (k, vals)
            for vals in itertools.product(*(
                [(k2, 0), (k2, 1)]
                for k2 in self.rpa[k]
                if type(k2) is str
            )))) for k in self.rpa}
        self.cond = {self.r[k]: list(itertools.chain.from_iterable(
            self.r[k2[0]] for k2 in self.rpa[k] if type(k2) is tuple))
            for k in self.rpa}

        v = list(self.r)
        pu = FactorizedDistribution(self.r.values(), cond=self.cond)
        f = {vi: lambda v, u, vi=vi, r=self.r, rpa=self.rpa:
             (T.cat([u[ui] for ui in self.r[vi]], dim=-1)
              .reshape((u[self.r[vi][0]].shape[0],)
                       + (2,) * len(self.r[vi][0][1]))[
                  (T.arange(u[self.r[vi][0]].shape[0]),)
                  + tuple(
                      (v[k] if isinstance(k, str) else u[k]).flatten()
                      for k, _ in self.r[vi][0][1]
                  )]).view(-1, 1) for vi in v}

        super().__init__(v, f, pu)

    def pmf(self, v, do={}, cond={}):
        pmf = T.exp(self.log_pmf(v, do, cond))
        return pmf if self.training else pmf.item()
        if cond:
            assert set(v).isdisjoint(cond)
            return (self.pmf(dict(v, **cond), do=do)
                    / self.pmf(cond, do=do))

        def _compare(v1, v2):
            return (all(v1[k] == v2[k].item() for k in v1))
        pmf = sum((T.exp(self.pu.log_pmf(u))
                   for u in self.pu.space()
                   if _compare(v, self(u=u, do=do))),
                  T.tensor(0.))
        return pmf if self.training else pmf.item()

    def log_pmf(self, v, do={}, cond={}):
        if cond:
            assert set(v).isdisjoint(cond)
            return (self.log_pmf(dict(v, **cond), do=do)
                    - self.log_pmf(cond, do=do))

        def _compare(v1, v2):
            return (all(v1[k] == v2[k].item() for k in v1))
        pmf = T.cat([self.pu.log_pmf(u)
                     for u in self.pu.space()
                     if _compare(v, self(u=u, do=do))], dim=-1)
        return T.logsumexp(pmf, dim=-1, keepdim=True)
