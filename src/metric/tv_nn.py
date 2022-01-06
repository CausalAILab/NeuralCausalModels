import sys

import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn

from src import metric
from src.metric import metrics
from src.scm.nn.simple import Simple


class LikelihoodEstimator(nn.Module):
    def __init__(self, v_sizes, device=None):
        super().__init__()
        self.v = sorted(list(v_sizes.keys()))
        self.v_sizes = v_sizes
        self.v_total_sizes = 0
        for v in self.v:
            self.v_total_sizes += self.v_sizes[v]
        self.model = Simple(v_size={}, u_size={}, o_size=self.v_total_sizes)

        if device is None:
            self.device = "cpu"
        else:
            self.device = device
            self.to(device)

    def forward(self, n=None, v=None):
        if v is not None:
            out = self.model({}, {}, v=v)
        else:
            out = self.model({}, {}, n=n).detach().cpu().numpy()
            out_pd = pd.DataFrame(out, columns=[f'{k}{i}' for k in self.v for i in range(self.v_sizes[k])])
            out = {k: T.FloatTensor(out_pd[[f'{k}{i}' for i in range(self.v_sizes[k])]].values) for k in self.v}
        return out

    def fit(self, v_dat, lr=0.0001, max_epochs=2000, patience=50, verbose=False):
        opt = T.optim.Adam(self.parameters(), lr=lr)
        nlpv = metric.probability_table(dat=v_dat)
        nlpv['_nlpv'] = -np.log(nlpv['P(V)'].astype(np.float32))

        #vs = T.FloatTensor(nlpv[[f'{k}{i}' for k in self.v for i in range(self.v_sizes[k])]].values).to(self.device)
        vs = T.FloatTensor(nlpv[self.v].values).to(self.device)
        nlpvs = T.FloatTensor(nlpv._nlpv.values).to(self.device)

        last_improvement = 0
        lowest_loss = None
        best_params = None
        epoch = 0
        while epoch < max_epochs and last_improvement < patience:
            epoch += 1
            last_improvement += 1

            opt.zero_grad()
            total_loss = 0
            for v, nlpv in zip(vs, nlpvs):
                nll = -self(v=v)
                true_loss = (T.exp(-nlpv) * (nll - nlpv)).mean(dim=0)
                total_loss += true_loss.item()
                true_loss.backward()
                del nll, true_loss
            if lowest_loss is None or total_loss < lowest_loss:
                lowest_loss = total_loss
                best_params = self.state_dict()
                last_improvement = 0
            if verbose and epoch % 50 == 49:
                print("{}: {}".format(epoch + 1, total_loss))
            opt.step()
        self.load_state_dict(best_params)

    def tv(self, n=100000):
        with T.no_grad():
            v_samples = self(n=n)
            return metrics.tv(dat=v_samples)


def naive_nn_tv(dat, device=None, return_kl=False, true_model=None):
    v_sizes = {k: dat[k].shape[1] for k in dat}
    naive_nn = LikelihoodEstimator(v_sizes, device=device)
    naive_nn.fit(dat, lr=0.001)
    if return_kl:
        naive_nn_kl = metrics.kl(true_model, naive_nn)
        return naive_nn.tv(n=1000000), naive_nn_kl
    return naive_nn.tv(n=1000000)


if __name__ == "__main__":
    if T.cuda.is_available():
        dev = T.device('cuda:0')
    else:
        dev = "cpu"

    d = sys.argv[1]

    print('loading data')
    dat = T.load(f'{d}/dat.th')

    print('computing naive NN TV')
    print(naive_nn_tv(dat, device=dev))
