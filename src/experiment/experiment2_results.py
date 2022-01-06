import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

cap = int(sys.argv[1]) if len(sys.argv) > 1 else None

sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 24})

records = []
for d in tqdm(glob.glob('out/BiasedNLLNCMPipeline/*')):
    try:
        record = {}
        record['key'] = os.path.basename(d)
        record.update(map(lambda t: tuple(t.split('=')), record['key'].split('-')))
        if cap is not None and int(record['trial_index']) >= cap:
            continue
        if not os.path.isfile(f'{d}/ate_results.json'):
            print("{} has no results".format(d))
            continue
        with open(f'{d}/ate_results.json') as file:
            record.update(json.load(file))
        with open(f'{d}/results.json') as file:
            old_results = json.load(file)
            record['true_tv'] = old_results['true_tv']
            record['err_ncm_tv'] = old_results['err_ncm_tv']
            record['kl'] = old_results['kl']
        records.append(record)
    except Exception as e:
        print(d, e)

df = pd.DataFrame.from_records(records)
df['n_samples'] = df['n_samples'].astype(int)

cols = ['err_naive_nn_ate', 'err_ncm_ate', 'err_werm_ate']
for col in cols:
    df[col] = df[col].astype(float).abs()
df['err_ncm_tv'] = df['err_ncm_tv'].astype(float).abs()
df['kl'] = df['kl'].astype(float).abs()
df['err_naive_tv'] = abs(df['naive_nn_ate'] - df['true_tv'])


order = ["backdoor", "frontdoor", "m", "napkin"]
fig, axes = plt.subplots(2, len(order), sharex=True, sharey='row')
for g_ind, graph in enumerate(order):
    ax_tv = axes[0][g_ind]
    ax_ate = axes[1][g_ind]

    df2 = (df.query(f'graph == "{graph}"')
           .rename(lambda s: str(s).replace('err_', 'mae_'), axis=1))
    melt_ate = df2.melt('n_samples', [col.replace('err_', 'mae_') for col in cols],
                    var_name='estimator', value_name='mae')
    sns_ate_ax = sns.lineplot(data=melt_ate, x='n_samples', y='mae', hue='estimator', marker='o', ax=ax_ate)

    melt_tv = df2.melt('n_samples', ['naive_nn_kl', 'kl'],
                           var_name='estimator', value_name='kl_val')
    sns_tv_ax = sns.lineplot(data=melt_tv, x='n_samples', y='kl_val', hue='estimator', marker='o', ax=ax_tv)

    ax_ate.set_xlabel("")
    ax_ate.get_legend().remove()
    ax_tv.get_legend().remove()

axes[0][0].set_xscale("log")
axes[0][0].set_yscale("log")
axes[1][0].set_yscale("log")

axes[0][0].set_xticks([1e3, 1e4, 1e5, 1e6])

axes[0][0].set_ylabel("KL of P(V)")
axes[1][0].set_ylabel("MAE of ATE")

trans = mtrans.blended_transform_factory(fig.transFigure,
                                             mtrans.IdentityTransform())
xlab = fig.text(.5, 20, "Number of Training Samples (n)", ha='center', fontsize=24)
xlab.set_transform(trans)

os.makedirs('img', exist_ok=True)
fig.set_figheight(10)
fig.set_figwidth(21)
fig.tight_layout()
fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.1)
fig.savefig(f'img/est_results.png', dpi=300, bbox_inches='tight')
