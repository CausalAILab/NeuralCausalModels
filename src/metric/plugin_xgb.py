import sys
import numpy as np
import torch as T
import pandas as pd
from src.metric.xgb_util import XGBProb, CountingProb


def plugin_xgb_ate(dat, cg_file):
    df = pd.DataFrame({k: v.numpy().flatten() for k, v in dat.items()}).reset_index()
    df['*'] = 0
    graph = cg_file.split('/')[-1].split('.')[0]

    def diff(g):
        return (-next(g) + next(g))

    if graph in ('simple', 'm'):
        pygx = XGBProb('Y', 'X')
        pygx.fit(df)
        return diff(pygx.predict({'Y': 1}, {'X': x}) for x in (0, 1))

    if graph == 'backdoor':
        pygxz = XGBProb('Y', ['X', 'Z'])
        pygxz.fit(df)
        pz = CountingProb('Z')
        pz.fit(df)

        def pydox(y, x):
            result = 0
            for z in df['Z'].unique():
                result += pygxz.predict({'Y': y}, {'X': x, 'Z': z}) * pz.predict({'Z': z})
            return result

        return diff(pydox(1, x) for x in (0, 1))

    if graph == 'frontdoor':
        pmgx = XGBProb('M', 'X')
        pmgx.fit(df)
        pygmx = XGBProb('Y', ['M', 'X'])
        pygmx.fit(df)
        px = CountingProb('X')
        px.fit(df)

        def pydox(y, x):
            result = 0
            for m in df['M'].unique():
                pydom = 0
                for x2 in df['X'].unique():
                    pydom += pygmx.predict({'Y': y}, {'M': m, 'X': x2}) * px.predict({'X': x2})
                pydom *= pmgx.predict({'M': m}, {'X': x})
                result += pydom
            return result

        return diff(pydox(1, x) for x in (0, 1))

    if graph == 'napkin':
        pyxgzw = XGBProb(['Y', 'X'], ['Z', 'W'])
        pyxgzw.fit(df)
        pxgzw = XGBProb('X', ['Z', 'W'])
        pxgzw.fit(df)
        pw = CountingProb('W')
        pw.fit(df)

        z_choice = df['Z'].mode().iat[0]

        def pydox(y, x):
            pyxdoz = 0
            pxdoz = 0
            for w in df['W'].unique():
                pyxdoz += pyxgzw.predict({'Y': y, 'X': x}, {'Z': z_choice, 'W': w}) * pw.predict({'W': w})
                pxdoz += pxgzw.predict({'X': x}, {'Z': z_choice, 'W': w}) * pw.predict({'W': w})
            return pyxdoz / pxdoz

        return diff(pydox(1, x) for x in (0, 1))

    return np.nan


if __name__ == '__main__':
    d = sys.argv[1]

    print('loading data')
    dat = T.load(f'{d}/dat.th')
    graph = d.split('\\')[-1].split('/')[-1].split('-')[0].split('=')[1]
    print("Graph = {}".format(graph))

    print('computing plugin ate')
    print(plugin_xgb_ate(dat, graph))
