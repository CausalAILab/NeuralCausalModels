import numpy as np
import torch

from src.scm import SCM
from src.ds.causal_graph import CausalGraph
from src.scm.distribution import BernoulliDistribution


class XORModel(SCM):
    def __init__(self, cg, dim=1, p=0.5, seed=None):
        self.cg = cg
        self.dim = dim
        self.p = p

        sizes = dict()
        for V in cg.v:
            if V == 'X' or V == 'Y':
                sizes[V] = 1
            else:
                sizes[V] = dim

        self.confounders = {V: [] for V in self.cg.v}
        for V1, V2 in cg.bi:
            conf_name = "U_{}{}".format(V1, V2)
            self.confounders[V1].append(conf_name)
            self.confounders[V2].append(conf_name)
            sizes[conf_name] = 1

        super().__init__(
            v=list(cg),
            f={V: self.get_xor_func(V) for V in cg},
            pu=BernoulliDistribution(list(sizes.keys()), sizes, p=p, seed=seed))

    def get_xor_func(self, V):
        conf_list = self.confounders[V]
        par_list = self.cg.pa[V]

        def xor_func(v, u):
            values = u[V]

            for conf in conf_list:
                values = torch.bitwise_xor(values, u[conf])
            for par in par_list:
                par_samp = v[par].long()
                if values.shape[1] >= par_samp.shape[1]:
                    values = torch.bitwise_xor(values, par_samp)
                else:
                    par_samp = torch.unsqueeze(torch.remainder(torch.sum(par_samp, 1), 2), 1)
                    values = torch.bitwise_xor(values, par_samp)

            return values

        return xor_func


if __name__ == "__main__":
    cg = CausalGraph.read("../../dat/cg/frontdoor.cg")
    m = XORModel(cg, dim=4, p=0.1)
    print(m(10, do={'X': torch.ones((10, 1), dtype=int)}))
