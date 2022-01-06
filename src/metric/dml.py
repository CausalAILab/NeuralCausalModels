import functools
import sys
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch as T
from sklearn.model_selection import train_test_split

from src.metric.xgb_util import XGBProb, CountingProb


def dml_ate(dat, cg_file, use_xgb=False):
    df = pd.DataFrame({k: v.numpy().flatten() for k, v in dat.items()}).reset_index()
    df['*'] = 0

    if use_xgb:
        temp_df_1, temp_df_2 = train_test_split(df, test_size=0.5)
        df_1 = temp_df_1.copy(deep=True)
        df_2 = temp_df_2.copy(deep=True)
        model_dict = dict()

    graph = cg_file.split('/')[-1].split('.')[0]
    def diff(g): return (-next(g) + next(g))

    def uif(x: Union[str, List[str]],
            z: Union[str, List[str]],
            y: Union[str, List[str]],
            x_val: Union[int, Dict[str, int]],
            y_val: Union[int, Dict[str, int]]):
        """ Compute uncentered influence function per datapoint for mSBD expression. """
        nonlocal df

        # argument validation and preprocessing

        # convert x, y, z to lists
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]
        if isinstance(z, str):
            z = [z]

        # confirm no overlap between x, y, and z
        if not (set(x).isdisjoint(y)
                and set(y).isdisjoint(z)
                and set(z).isdisjoint(x)):
            raise ValueError(f'x, y, z not all disjoint: {x}, {y}, {z}')

        # convert x_val, y_val to dicts
        if not isinstance(x_val, dict):
            if len(x) != 1:
                raise ValueError('x must be a singleton if x_val is a number')
            x_val = {x[0]: x_val}
        if not isinstance(y_val, dict):
            if len(y) != 1:
                raise ValueError('y must be a singleton if y_val is a number')
            y_val = {y[0]: y_val}

        # confirm x_val, y_val match with x, y
        if not set(x_val) == set(x):
            raise ValueError(f'mismatched x and x_val: {x}, {x_val}')
        if not set(y_val) == set(y):
            raise ValueError(f'mismatched y and y_val: {y}, {y_val}')

        # sort x, y, z for consistency
        x = sorted(x)
        y = sorted(y)
        z = sorted(z)

        # helper strings
        xs = ', '.join(x)
        zs = ', '.join(z)
        xzs = ', '.join(sorted(x + z))
        xyzs = ', '.join(sorted(x + y + z))
        xvs = ', '.join(f'{k} = {v}' for k, v in sorted(x_val.items()))
        yvs = ', '.join(f'{k} = {v}' for k, v in sorted(y_val.items()))
        xyvs = ', '.join(f'{k} = {v}' for k, v in sorted(list(x_val.items())
                                                         + list(y_val.items())))
        uifs = 'uif(%s; %s; %s; %s; %s)' % (x, z, y, x_val, y_val)

        # df column names for caching
        px = 'P(%s)' % xs
        pz = 'P(%s)' % zs
        pxz = 'P(%s)' % xzs
        pxyz = 'P(%s)' % xyzs
        if len(z) != 0:
            pyvxz = 'P(%s, %s)' % (yvs, xzs)
            pxvz = 'P(%s, %s)' % (xvs, zs)
            pyvxvz = 'P(%s, %s)' % (xyvs, zs)
        else:
            pyvxz = 'P(%s)' % (yvs)
            pxvz = 'P(%s)' % (xvs)
            pyvxvz = 'P(%s)' % (xyvs)

        # conditional expressions
        pyvgxz = 'P(%s | %s)' % (yvs, xzs)
        if len(z) != 0:
            pxgz = 'P(%s | %s)' % (xs, zs)
            pyvgxvz = 'P(%s | %s, %s)' % (yvs, xvs, zs)
        else:
            pxgz = 'P(%s)' % xs
            pyvgxvz = 'P(%s | %s)' % (yvs, xvs)

        if not use_xgb:
            if px not in df:
                df = pd.merge(df,
                              (df.groupby(x)['*'].count() / len(df))
                              .rename(px).to_frame().reset_index())
            if len(z) != 0 and pz not in df:
                df = pd.merge(df,
                              (df.groupby(z)['*'].count() / len(df))
                              .rename(pz).to_frame().reset_index())
            if pxz not in df:
                df = pd.merge(df,
                              (df.groupby(sorted(x + z))['*'].count() / len(df))
                              .rename(pxz).to_frame().reset_index())
            if pxyz not in df:
                df = pd.merge(df,
                              (df.groupby(sorted(x + y + z))['*'].count() / len(df))
                              .rename(pxyz).to_frame().reset_index())

            if pyvxz not in df:
                df = pd.merge(df,
                              (df.groupby(sorted(x + y + z))['*'].count() / len(df))
                              .rename(pyvxz).to_frame().reset_index()
                              .query(' and '.join(f'{k} == {v}' for k, v in y_val.items()))
                              .drop(list(y_val), axis=1))

            if pxvz not in df:
                result = (df.groupby(sorted(x + z))['*'].count() / len(df)) \
                                  .rename(pxvz).to_frame().reset_index() \
                                  .query(' and '.join(f'{k} == {v}' for k, v in x_val.items())) \
                                  .drop(list(x_val), axis=1)
                if len(z) != 0:
                    df = pd.merge(df, result)
                else:
                    df[pxvz] = result.iloc[0][pxvz]

            if pyvxvz not in df:
                result = (df.groupby(sorted(x + y + z))['*'].count() / len(df)) \
                              .rename(pyvxvz).to_frame().reset_index() \
                              .query(' and '.join(f'{k} == {v}' for k, v in (list(x_val.items())
                                                                             + list(y_val.items())))) \
                              .drop(list(x_val) + list(y_val), axis=1)
                if len(z) != 0:
                    df = pd.merge(df, result)
                else:
                    df[pyvxvz] = result.iloc[0][pyvxvz]

            if pxgz not in df:
                if len(z) != 0:
                    df[pxgz] = df[pxz] / df[pz]
                else:
                    df[pxgz] = df[px]

            if pyvgxz not in df:
                df[pyvgxz] = df[pyvxz] / df[pxz]

            if pyvgxvz not in df:
                df[pyvgxvz] = df[pyvxvz] / df[pxvz]

        else:
            if pxgz not in model_dict:
                if len(z) != 0:
                    pxgz_m1 = XGBProb(x, z)
                    pxgz_m2 = XGBProb(x, z)
                else:
                    pxgz_m1 = CountingProb(x)
                    pxgz_m2 = CountingProb(x)
                pxgz_m1.fit(df_1)
                pxgz_m2.fit(df_2)
                model_dict[pxgz] = (pxgz_m1, pxgz_m2)

            if pyvgxz not in model_dict:
                pygxz_m1 = XGBProb(y, x + z)
                pygxz_m2 = XGBProb(y, x + z)
                pygxz_m1.fit(df_1)
                pygxz_m2.fit(df_2)
                model_dict[pyvgxz] = (pygxz_m1, pygxz_m2)

            if pxgz not in df_1:
                pxgz_m1, pxgz_m2 = model_dict[pxgz]
                df_1[pxgz] = pxgz_m2.predict_dat(df_1)
                df_2[pxgz] = pxgz_m1.predict_dat(df_2)

            if pyvgxz not in df_1:
                pygxz_m1, pygxz_m2 = model_dict[pyvgxz]
                df_1[pyvgxz] = pygxz_m2.predict_dat(df_1, y_fix=y_val)
                df_2[pyvgxz] = pygxz_m1.predict_dat(df_2, y_fix=y_val)

            if pyvgxvz not in df_1:
                pygxz_m1, pygxz_m2 = model_dict[pyvgxz]
                df_1[pyvgxvz] = pygxz_m2.predict_dat(df_1, y_fix=y_val, x_fix=x_val)
                df_2[pyvgxvz] = pygxz_m1.predict_dat(df_2, y_fix=y_val, x_fix=x_val)

            df = pd.concat([df_1, df_2])

        if uifs not in df:
            df[uifs] = (
                functools.reduce(lambda a, k: a & (df[k] == x_val[k]), x_val, 1)
                / df[pxgz]
                * (functools.reduce(lambda a, k: a & (df[k] == y_val[k]), y_val, 1)
                   - df[pyvgxz])
                + df[pyvgxvz]
            )

        df.to_csv('tmp.csv', sep='\t', index=False, float_format='%.5f')
        return df[uifs]

    def cif(x: Union[str, List[str]],
            z: Union[str, List[str]],
            y: Union[str, List[str]],
            x_val: Union[int, Dict[str, int]],
            y_val: Union[int, Dict[str, int]]):
        """ Compute (centered) influence function per datapoint for mSBD expression. """
        nonlocal df

        u = uif(x, z, y, x_val, y_val)
        ifs = u.name[1:]
        if ifs not in df:
            df[ifs] = u - u.mean()
        return df[ifs]

    if graph in ('simple', 'm'):
        return diff(uif('X', [], 'Y', x, 1).mean() for x in (0, 1))
        #return diff(T.tensor(len(df.query(f'Y == 1 and X == {x}')) / len(df.query(f'X == {x}')))
        #            for x in (0, 1))
    elif graph == 'backdoor':
        return diff(uif('X', 'Z', 'Y', x, 1).mean() for x in (0, 1))
    elif graph == 'frontdoor':
        results = [0, 0]
        unique_ms = df['M'].unique()
        for x in (0, 1):
            for m in unique_ms:
                pydom = uif('M', 'X', 'Y', m, 1)
                pmdox = uif('X', [], 'M', x, m)
                pmdox_cent = cif('X', [], 'M', x, m)
                results[x] += ((pydom * pmdox).mean() + (pmdox_cent * pydom).mean())
        return results[1] - results[0]
    elif graph == 'napkin':
        zval = df['Z'].mode().iat[0]
        pyx0dor = uif('Z', 'W', ['X', 'Y'], {'Z': zval}, {'X': 0, 'Y': 1})
        px0dor = uif('Z', 'W', 'X', zval, 0)
        pyx1dor = uif('Z', 'W', ['X', 'Y'], {'Z': zval}, {'X': 1, 'Y': 1})
        px1dor = uif('Z', 'W', 'X', zval, 1)
        return (pyx1dor.mean() / px1dor.mean()) - (pyx0dor.mean() / px0dor.mean())
    else:
        return np.nan


if __name__ == '__main__':
    d = sys.argv[1]

    print('loading data')
    dat = T.load(f'{d}/dat.th')
    graph = d.split('\\')[-1].split('/')[-1].split('-')[0].split('=')[1]
    print("Graph = {}".format(graph))

    print('computing dml ate')
    print(dml_ate(dat, graph, use_xgb=True))
