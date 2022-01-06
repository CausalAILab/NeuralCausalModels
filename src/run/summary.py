import glob
import json
import os
import sys

import pandas as pd
from tqdm.auto import tqdm

os.chdir(os.path.dirname(__file__) + '/../..')


def process(d):
    results = {}
    _, results['name'], results['params'] = d.rsplit('/', 2)
    try:
        results.update(dict(tuple(t.split('=') for t in results['params'].split('-'))))
        assert results['graph'].strip() != ''
    except Exception as e:
        print(d, e)
        return {}
    results['n_samples'] = int(results['n_samples'])

    try:
        with open(f'{d}/results.json') as file:
            results.update(json.load(file))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
    return results


df = pd.DataFrame.from_records(list(map(process,
                                        filter(lambda s: not s.startswith('out/logs'),
                                               tqdm(glob.glob('out/*/*'))))))
err_cols = df.columns[df.columns.str.contains('err')]
mae_cols = err_cols.str.replace('err', 'mae')
t = (
    df
    .groupby(['name', 'graph', 'n_samples'])[err_cols]
    .apply(lambda s: s.abs().mean())
    .rename(dict(zip(err_cols, mae_cols)), axis=1)
    .assign(n_trials=df.groupby(['name', 'graph', 'n_samples'])
            .apply(lambda s: len(s.err_ncm_ate.dropna()))))
t[~t.mae_ncm_ate.isna()].to_csv('out/summary.csv', float_format='%.5f')
