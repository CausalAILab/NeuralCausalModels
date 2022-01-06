import sys
import os
from time import time

import numpy as np
import pandas as pd
import torch as T
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from .metrics import tv


def train_xgb(x, y, regval, binary=True, weight=None):
    return (XGBRegressor(n_estimators=20, max_depth=10,
                         reg_alpha=regval/2, reg_lambda=regval,
                         objective='binary:logistic' if binary else 'reg:squarederror',
                         eval_metric='logloss' if binary else 'mae',
                         nthread=-1)
            .fit(x, y, sample_weight=weight))


def learn_hyperparameter(regvals, x, y, binary=True, timer=None):
    t0 = time()
    trX, teX, trY, teY = train_test_split(x, y, test_size=1/3)
    if binary:
        def error(regval):
            return ((train_xgb(trX, trY, regval, binary)
                     .predict(teX) >= 0.5).astype(int) != np.array(teY).flatten()).sum()
    else:
        def error(regval):
            return np.abs(train_xgb(trX, trY, regval, binary)
                          .predict(teX) - np.array(teY).flatten()).sum()
    result = min(tqdm(regvals), key=error)
    t_tot = time() - t0
    if timer is not None:
        timer[0] += t_tot
    return result


def learn_w(w_hat, x, regval, timer=None):
    t0 = time()
    w_model = train_xgb(x, w_hat, regval, binary=False)
    t_tot = time() - t0
    if timer is not None:
        timer[0] += t_tot
    w = w_model.predict(x)
    w[w < 0] = np.random.rand(len(w[w < 0])) / np.abs(w).min()
    return w


def werm(trX, teX, y, w, lambda_h, binary=True, timer=None):
    t0 = time()
    model = train_xgb(trX, y, lambda_h, binary, weight=w)
    t_tot = time() - t0
    if timer is not None:
        timer[0] += t_tot
    return model.predict(teX).mean()


def werm_ate(dat, cg_file, n=100000, regvals=None, skip_train=False):
    df = pd.DataFrame({k: (v * (1 << np.arange(v.shape[-1]))).sum(dim=1) for k, v in dat.items()})
    df['*'] = 0
    df_full = pd.DataFrame({f'{k}{i}': v[:, i].numpy()
                            for k, v in dat.items()
                            for i in range(dat[k].shape[-1])})

    if regvals is None:
        regvals = np.arange(0, 10 + 0.2, 0.2)
    graph = cg_file.split('/')[-1].split('.')[0]
    def diff(g): return (-next(g) + next(g)).item()
    def key(k): return [f'{k}{i}' for i in range(dat[k].shape[-1])]

    timer = None
    if skip_train:
        timer = [0]
    ate_result = float('nan')
    if graph == 'backdoor':  # W = P(x) / P(x|z)
        df = pd.merge(df,
                      (df.groupby(['X'])['*'].count().rename('P(X)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['Z'])['*'].count().rename('P(Z)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['X', 'Z'])[
                       '*'].count().rename('P(X, Z)') / len(df))
                      .to_frame().reset_index())
        df['P(X | Z)'] = df['P(X, Z)'] / df['P(Z)']
        w_hat = (df['P(X)'] / df['P(X | Z)']).values
        """
        lambda_w = learn_hyperparameter(
            regvals, df_full[key('X') + key('Z')], w_hat,
            binary=False, timer=timer)
        w = learn_w(w_hat, df_full[key('X') + key('Z')],
                    lambda_w, timer=timer)
        """
        lambda_w = learn_hyperparameter(
            regvals, df[['X', 'Z']], w_hat,
            binary=False, timer=timer)
        w = learn_w(w_hat, df[['X', 'Z']],
                    lambda_w, timer=timer)
        lambda_h = learn_hyperparameter(
            regvals, df[['X']], df['Y'], binary=True, timer=timer)
        print(lambda_w, lambda_h)
        ate_result = diff(werm(df[['X']], df[['X']].assign(X=x), df['Y'],
                               w, lambda_h, binary=True, timer=timer)
                          for x in (0, 1))
    elif graph == 'frontdoor':  # W = P^{Wy}(y|m)/P(y|m,x); Wy = P(m)/P(m|x)
        df = pd.merge(df,
                      (df.groupby(['X'])['*'].count().rename('P(X)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['M'])['*'].count().rename('P(M)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['X', 'M'])[
                       '*'].count().rename('P(X, M)') / len(df))
                      .to_frame().reset_index())
        df['P(M | X)'] = df['P(X, M)'] / df['P(X)']
        df['Wy'] = df['P(M)'] / df['P(M | X)']
        # P^{Wy}(y|m) = P^{Wy}(y|m) = P^{Wy}(y, m) / P^{Wy}(m)
        df = pd.merge(df,
                      (df.groupby(['M', 'X', 'Y'])[
                       '*'].count().rename('P(V)') / len(df))
                      .to_frame().reset_index())
        df['P^{Wy}(V)'] = df['Wy'] * df['P(V)']
        assert (df.groupby(['M', 'X', 'Y'])['P^{Wy}(V)']
                .apply(lambda s: len(s) in (0, 1) or len(s.drop_duplicates(keep=False)) == 0).all())
        df = pd.merge(df,
                      ((df.groupby(['M', 'X', 'Y'])[
                       'P^{Wy}(V)'].first().reset_index()))
                      .groupby(['Y', 'M'])['P^{Wy}(V)'].sum().rename('P^{Wy}(Y, M)')
                      .to_frame().reset_index())
        df = pd.merge(df,
                      ((df.groupby(['M', 'X', 'Y'])[
                       'P^{Wy}(V)'].first().reset_index()))
                      .groupby(['M'])['P^{Wy}(V)'].sum().rename('P^{Wy}(M)')
                      .to_frame().reset_index())
        df['P^{Wy}(Y | M)'] = df['P^{Wy}(Y, M)'] / df['P^{Wy}(M)']
        df = pd.merge(df,
                      (df.groupby(['X', 'M'])[
                       '*'].count().rename('P(X, M)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['Y', 'X', 'M'])[
                       '*'].count().rename('P(Y, X, M)') / len(df))
                      .to_frame().reset_index())
        df['P(Y | M, X)'] = df['P(Y, X, M)'] / df['P(X, M)']
        w_hat = (df['P^{Wy}(Y | M)'] / df['P(Y | M, X)']).values
        lambda_w = learn_hyperparameter(
            regvals, df[['X', 'Y']], w_hat, binary=False, timer=timer)
        w = learn_w(w_hat, df[['X', 'Y']], lambda_w, timer=timer)
        lambda_h = learn_hyperparameter(
            regvals, df[['X']], df['Y'], binary=True, timer=timer)
        ate_result = diff(werm(df[['X']], df[['X']].assign(X=x), df['Y'],
                               w, lambda_h, binary=True, timer=timer)
                          for x in (0, 1))
        print(lambda_w, lambda_h, ate_result)
    elif graph == 'napkin':  # W = P(z) / P(z|w)
        df = pd.merge(df,
                      (df.groupby(['Z'])['*'].count().rename('P(Z)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['W'])['*'].count().rename('P(W)') / len(df))
                      .to_frame().reset_index())
        df = pd.merge(df,
                      (df.groupby(['Z', 'W'])[
                       '*'].count().rename('P(Z, W)') / len(df))
                      .to_frame().reset_index())
        df['P(Z | W)'] = df['P(Z, W)'] / df['P(Z)']
        w_hat = (df['P(Z)'] / df['P(Z | W)']).values
        lambda_w = learn_hyperparameter(
            regvals, df[['W', 'Z']], w_hat, binary=False, timer=timer)
        w = learn_w(w_hat, df[['W', 'Z']], lambda_w, timer=timer)
        lambda_h = learn_hyperparameter(
            regvals, df[['X', 'Z']], df['Y'], binary=True, timer=timer)
        print(lambda_w, lambda_h)
        ate_result = diff(werm(df[['X', 'Z']], df[['X', 'Z']].assign(X=x), df['Y'],
                               w, lambda_h, binary=True, timer=timer)
                          for x in (0, 1))
    elif graph in ('m', 'simple'):
        ate_result = tv(dat=dat)

    if not skip_train:
        return ate_result
    else:
        return ate_result, timer[0]


if __name__ == '__main__':
    d = sys.argv[1]

    print('loading data')
    dat = T.load(f'{d}/dat.th')
    graph = next(s.split('=')[1]
                 for s in d.strip(os.path.sep).split(os.path.sep)[-1].split('-')
                 if s.split('=')[0] == 'graph')

    print('computing werm ate')
