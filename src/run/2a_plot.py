import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

cap = int(sys.argv[1]) if len(sys.argv) > 1 else None

sns.set_style('darkgrid')

records = []
for d in tqdm(glob.glob('out/BiasedNLLNCMPipeline/*')):
    if d.startswith('out/logs'):
        continue
    try:
        record = {}
        _, record['pipeline'], record['key'] = d.split('/')
        record.update(map(lambda t: tuple(t.split('=')), record['key'].split('-')))
        if cap is not None and int(record['trial_index']) >= cap:
            continue
        assert os.path.isfile(f'{d}/best.th')
        with open(f'{d}/ate_results.json') as file:
            record.update(json.load(file))
        records.append(record)
    except Exception as e:
        print(d, e)

df = pd.DataFrame.from_records(records)
df['n_samples'] = df['n_samples'].astype(int)

cols = ['err_plugin_ate', 'err_werm_ate', 'err_ncm_ate', 'err_dml_ate', 'err_naive_nn_ate']
for col in cols:
    df[col] = df[col].astype(float).abs()
for graph in tqdm(df.graph.unique()):
    plt.figure()
    plt.gcf()
    df2 = (df.query(f'graph == "{graph}"')
           .rename(lambda s: str(s).replace('err_', 'mae_'), axis=1))
    melt = df2.melt('n_samples', [col.replace('err_', 'mae_') for col in cols],
                    var_name='estimator', value_name='mae')
    g = sns.lineplot(data=melt, x='n_samples', y='mae', hue='estimator', marker='o')
    plt.xlabel("# samples")
    plt.ylabel("Mean absolute error")
    plt.title(f"Mean absolute error vs. # samples for {graph} graph")
    print(graph)
    print(melt.groupby(['n_samples', 'estimator']).mean().mae.to_frame().reset_index())
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    os.makedirs('img', exist_ok=True)
    plt.savefig(f'img/{graph}.png', dpi=300)
