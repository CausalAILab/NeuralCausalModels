import numpy as np
import torch as T
import torch.nn as nn

from .distribution import UniformDistribution
from .nn import Simple
from .scm import SCM


class NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, default_module=Simple):
        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        super().__init__(
            v=list(cg),
            f=nn.ModuleDict({
                k: f[k] if k in f else default_module(
                    {k: self.v_size[k] for k in self.cg.pa[k]},
                    {k: self.u_size[k] for k in self.cg.v2c2[k]},
                    self.v_size[k],
                )
                for k in cg}),
            pu=UniformDistribution(self.cg.c2))

    def biased_nll(self, v, n=1, do={}):
        assert not set(do.keys()).difference(self.v)
        mode = self.training
        try:
            self.train()
            batch_size = len(next(iter(v.values())))
            u = {k: t.expand((batch_size,) + tuple(t.shape)).transpose(0, 1)
                 for k, t in self.pu.sample(n=n).items()}  # (n, batch_size, var_size)
            v_new = {k: t.expand((n,) + t.shape).float()
                     for k, t in v.items()}
            logpv = 0
            for k in self.v:
                if k in do:
                    if do[k] != v[k]:
                        return float('-inf')
                else:
                    logpv += self.f[k](v_new, u, v_new[k])
            logpv = T.logsumexp(logpv, dim=0) - np.log(n)
            return -logpv
        finally:
            self.train(mode=mode)

    def nll(self, v, n=1, do={}, m=100000, alpha=80, return_biased=False):
        r"""
        Uses the SUMO / Russian roulette estimator to compute an unbiased estimate of NLL.

        Let $\hat L^m$ be a biased estimator of NLL:

        $$\hat L^n_b(x) := -\log(\sum_i^n P(x | u_i))$$

        where $u_{1:n} ~ P(U)$. Let

        $$\Delta_k(x) := \hat L^{k + 1}_b(x) - \hat L^k_b(x)$$.

        Clearly, for any integral $m > 0$, the infinite series

        $$S_m(x) = \{\hat L^m_b(x), \Delta_{m}(x), \Delta_{m+1}(x), \dots\}$$

        sums to -\log(P(x)). Our Russian roulette estimator follows:

        $$SUMO_m(x) := \hat L^m_b(x) + \sum_{k = m}^{K + m - 1} \frac{\Delta_k(x)}{P(K \geq k)}$$

        where $K ~ P(\mathcal K)$ for any positive distribution $P(\mathcal K)$ on the
        positive integers.

        In this function, we set:

        $$
        P(K \geq k) := \begin{cases}
            k^{-1}                         & k > \alpha
            \alpha^{-1} (0.9)^(k - \alpha) & k \geq \alpha
        \end{cases}
        $$

        with $\alpha = 80$.

        Parameters
        ----------
        v : dict, default={}
            Settings of V for which to estimate negative log likelihood.

        n : int, default=10
            Number of SUMO estimators to average over for each sample in v.

        do : dict, default={}
            Dict of variables to intervene on and their corresponding values.

        m : int, default=10000
            Minimum number of samples.

        alpha : int, default=80
            Shape parameter for $P(\mathcal K)$.

        return_biased : bool, default=False
            Whether or not to return (estimate, biased_estimate) as a tuple, or just the estimate.
        """
        assert not set(do.keys()).difference(self.v)
        mode = self.training
        try:
            self.train()
            batch_size = len(next(iter(v.values())))

            # sample n Ks per batch (batch_size, n)
            uk = np.random.rand(batch_size, n)
            K = np.where(uk > 1 / alpha,
                         np.floor(1 / uk),
                         np.floor(np.log(alpha * uk) / np.log(0.9) + alpha))

            # compute log probabilities (batch_size, max_sum_n_samples)
            n_samples = K + (m - 1)
            max_sum_n_samples = int(n_samples.sum(axis=1).max())
            u = {k: t.reshape((batch_size, max_sum_n_samples, t.shape[-1]))
                 for k, t in self.pu.sample(n=batch_size * max_sum_n_samples).items()}
            v_new = {k: t.expand((max_sum_n_samples,) + t.shape).transpose(0, 1)
                     for k, t in v.items()}

            logpv = 0
            for k in self.v:
                if k in do:
                    if do[k] != v[k]:
                        return float('-inf')
                else:
                    logpv += self.f[k](v_new, u, v_new[k])
            assert tuple(logpv.shape) == (batch_size, max_sum_n_samples), \
                (logpv.shape, batch_size, max_sum_n_samples)

            # compute weights (max_n_samples - m + 1,)
            ik = np.arange(K.max())
            ipk = T.tensor(np.where(ik < alpha, ik, alpha * 0.9 ** (alpha - ik)),
                           device=self.device_param.device)

            # compute SUMO given samples (batch_size, n)
            indices = np.pad(n_samples, [(0, 0), (1, 0)]).cumsum(axis=1).astype(int)
            assert (np.diff(indices) > 0).all()
            estimates = T.zeros(batch_size, n, device=self.device_param.device)
            for i in range(batch_size):
                for j, (s, e) in enumerate(zip(indices[i, :-1], indices[i, 1:])):
                    assert e - s == n_samples[i, j]
                    samples = logpv[i, s:e]
                    vals = (T.logcumsumexp(samples, dim=0)
                            - T.log(T.arange(len(samples), device=self.device_param.device) + 1))
                    vals = vals[m-1:]
                    estimates[i][j] = vals[0] + (T.diff(vals) * ipk[:len(vals)-1]).sum()

            # return empirical mean of SUMO estimates per sample (batch_size,)
            if return_biased:
                return (-estimates.mean(dim=1),
                        T.logsumexp(logpv.flatten(), dim=0) - T.log(T.tensor(logpv.numel())))
            else:
                return -estimates.mean(dim=1)
        finally:
            self.train(mode=mode)

    def nll_marg(self, v, n=1, m=10000, do={}, return_biased=False):
        assert not set(v.keys()).difference(self.v)
        assert not set(do.keys()).difference(self.v)

        marg_set = set(self.v).difference(v.keys()).difference(do.keys())
        marg_space = self.space(select=marg_set)
        pv = 0
        biased_pv = 0
        for marg_v in marg_space:
            v_joined = dict()
            v_joined.update(marg_v)
            v_joined.update(v)
            v_joined.update(do)
            #nll_all, biased_nll = self.nll(v_joined, n=n, m=m, do=do, return_biased=return_biased)
            nll_all = self.biased_nll(v_joined, n=m, do=do)
            biased_nll = 0
            pv += T.exp(-nll_all)
            if return_biased:
                biased_pv += T.exp(-biased_nll)
        if return_biased:
            return -T.log(pv), -T.log(biased_pv)
        return -T.log(pv)
