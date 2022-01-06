import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch as T

from src.scm import CTM
from src.metric.tv_nn import LikelihoodEstimator


@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    device = net.device_param.device
    try:
        net.cpu()
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()
        net.to(device)


def ate(m, n=1000000, cuda=False):
    with evaluating(m):
        if cuda:
            m = m.cuda()
        if isinstance(m, CTM):
            return (m.pmf({'Y': 1}, do={'X': T.tensor([[1]], )})
                    - m.pmf({'Y': 1}, do={'X': T.tensor([[0]])}))
        else:
            y1 = m(n, do={'X': T.ones(n, 1)})['Y']
            y0 = m(n, do={'X': T.zeros(n, 1)})['Y']
            return (y1.float().mean() - y0.float().mean()).item()


def tv(m=None, dat=None, n=1000000):
    if isinstance(m, CTM):
        with evaluating(m):
            return (m.pmf({'Y': 1}, cond={'X': 1})
                    - m.pmf({'Y': 1}, cond={'X': 0}))
    else:
        if dat is None:
            with evaluating(m):
                dat = m(n)
        return (dat['Y'][dat['X'] == 1].float().mean()
                - dat['Y'][dat['X'] == 0].float().mean()).item()


def probability_table(m=None, n=1000000, do={}, dat=None):
    if isinstance(m, CTM):
        with evaluating(m):
            return pd.DataFrame.from_records([
                {**v, 'P(V)': m.pmf(v, do=do)}
                for v in m.space(tensor=False)
            ])
    elif isinstance(m, LikelihoodEstimator):
        dat = m(n=n)
        return probability_table(n=n, dat=dat)
    else:
        if dat is None:
            with evaluating(m):
                dat = m(n, do=do)
        else:
            n = len(next(iter(dat.values())))
        df = pd.DataFrame({k: v.detach().flatten().numpy()
                           for k, v in dat.items()},
                          index=range(n))
        return (df.groupby(list(df.columns))
                .apply(lambda x: len(x) / len(df))
                .rename('P(V)').reset_index()
                [[*df.columns, 'P(V)']])


def kl(truth, ncm, n=1000000):
    m_table = probability_table(ncm, n=n)
    t_table = probability_table(truth, n=n)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()


def supremum_norm(truth, ncm, n=1000000):
    m_table = probability_table(ncm, n=n)
    t_table = probability_table(truth, n=n)
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='outer', on=cols, suffixes=['_t', '_m']).fillna(0.0)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_m - p_t).abs().max()


def plugin_ate(dat, cg_file):
    if not isinstance(cg_file, str):
        return float('nan')
    graph = cg_file.split('/')[-1].split('.')[0]
    def diff(g): return (-next(g) + next(g)).item()
    if graph == 'backdoor':
        return diff(
            sum(dat['Y'][(dat['Z'] == z)
                         & (dat['X'] == x)].float().mean()
                * (dat['Z'] == z).float().mean()
                for z in (0, 1))
            for x in (0, 1))
    elif graph == 'frontdoor':
        return diff(
            sum((dat['M'][dat['X'] == x] == m).float().mean()
                * sum(dat['Y'][(dat['X'] == xp)
                               & (dat['M'] == m)].float().mean()
                      * (dat['X'] == xp).float().mean()
                      for xp in (0, 1))
                for m in (0, 1))
            for x in (0, 1))
    elif graph == 'napkin':
        return diff(
            sum(sum((((dat['X'] == x)
                      & (dat['Y'] == 1))
                     [(dat['W'] == w)
                      & (dat['Z'] == z)]).float().mean()
                    * (dat['W'] == w).float().mean()
                    for w in (0, 1))
                / sum(((dat['X'] == x)
                       [(dat['W'] == w)
                        & (dat['Z'] == z)]).float().mean()
                      * (dat['W'] == w).float().mean()
                      for w in (0, 1))
                for z in (0, 1)) / 2
            for x in (0, 1))
    elif graph in ('m', 'simple'):
        return tv(dat=dat)
    else:
        return float('nan')


def all_metrics(truth, ncm, dat, cg_file, n=1000000):
    m = dict(
        true_ate=ate(truth),
        ncm_ate=ate(ncm, n=n),
        true_tv=tv(truth),
        ncm_tv=tv(ncm, n=n),
        kl=kl(truth, ncm, n=n),
        supremum_norm=supremum_norm(truth, ncm, n=n),
        plugin_ate=plugin_ate(dat, cg_file))
    m['dat_tv'] = (dat['Y'][dat['X'] == 1].float().mean()
                   - dat['Y'][dat['X'] == 0].float().mean()).item()

    m['err_ncm_ate'] = m['true_ate'] - m['ncm_ate']
    m['err_plugin_ate'] = m['true_ate'] - m['plugin_ate']
    m['err_ncm_tv'] = m['true_tv'] - m['ncm_tv']
    m['err_dat_tv'] = m['true_tv'] - m['dat_tv']
    m['err_dat_tv_ncm_tv'] = m['dat_tv'] - m['ncm_tv']
    m['err_plugin_ate_ncm_ate'] = m['plugin_ate'] - m['ncm_ate']
    return m


def all_metrics_minmax(truth, ncm_min, ncm_max, dat, cg_file, n=1000000):
    m = dict(
        true_ate=ate(truth),
        ncm_min_ate=ate(ncm_min, n=n),
        ncm_max_ate=ate(ncm_max, n=n),
        true_tv=tv(truth),
        ncm_min_tv=tv(ncm_min, n=n),
        ncm_max_tv=tv(ncm_max, n=n),
        kl_min=kl(truth, ncm_min, n=n),
        kl_max=kl(truth, ncm_max, n=n),
        supremum_norm_min=supremum_norm(truth, ncm_min, n=n),
        supremum_norm_max=supremum_norm(truth, ncm_max, n=n),
        plugin_ate=plugin_ate(dat, cg_file))
    m['dat_tv'] = (dat['Y'][dat['X'] == 1].float().mean()
                   - dat['Y'][dat['X'] == 0].float().mean()).item()

    m['err_ncm_min_ate'] = m['true_ate'] - m['ncm_min_ate']
    m['err_ncm_max_ate'] = m['true_ate'] - m['ncm_max_ate']
    m['err_plugin_ate'] = m['true_ate'] - m['plugin_ate']
    m['err_ncm_min_tv'] = m['true_tv'] - m['ncm_min_tv']
    m['err_ncm_max_tv'] = m['true_tv'] - m['ncm_max_tv']
    m['err_dat_tv'] = m['true_tv'] - m['dat_tv']
    m['err_dat_tv_ncm_min_tv'] = m['dat_tv'] - m['ncm_min_tv']
    m['err_dat_tv_ncm_max_tv'] = m['dat_tv'] - m['ncm_max_tv']
    m['err_plugin_ate_ncm_min_ate'] = m['plugin_ate'] - m['ncm_min_ate']
    m['err_plugin_ate_ncm_max_ate'] = m['plugin_ate'] - m['ncm_max_ate']
    m['minmax_ate_gap'] = m['ncm_max_ate'] - m['ncm_min_ate']
    m['minmax_tv_gap'] = m['ncm_max_tv'] - m['ncm_min_tv']
    return m


if __name__ == '__main__':
    d = sys.argv[1]

    print('loading data')
    dat = T.load(f'{d}/dat.th')
    graph = next(s.split('=')[1]
                 for s in d.strip('/').split('/')[-1].split('-')
                 if s.split('=')[0] == 'graph')

    print('computing plugin ate')
    print(plugin_ate(dat, graph))
