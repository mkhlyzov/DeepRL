import sys
import json
import pathlib
import time

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import clear_output
#%%
ROOT = list(pathlib.Path.home().glob('*/DeepRL/'))[0]
logs_dir = 'logs/LunarLander'
# columns = [
#     'env_steps', 'optimization_steps', 'train_score', 'eval_score',
#     'time_per_env_step', 'episodes', 'loss', 'grad_norm',
#     'weight_norm', 'features_dot', 'features_cos'
# ]
my_dirs = [
    # 'Adam_lr=1e-4',
    'Adam_lr=3e-4',
    # 'Adam_lr=1e-3'
]
index = 'env_steps'
col_name = 'train_score'

def mask_index(idx):
    return idx[idx < 4_000_000]

plt.style.use('seaborn')
fig, ax = plt.subplots(dpi=500)

for d in ROOT.joinpath(logs_dir).iterdir():
    if not d.is_dir() or d.name not in my_dirs:
        continue
    df_list = [
        pd.read_csv(file, index_col=(index))
        for file in d.glob('*.csv')
    ]
    if not df_list:
        continue
    
    mean = pd.concat(df_list).groupby([index]).mean()
    std = pd.concat(df_list).groupby([index]).std()
    
    mean_ = mean[col_name][mask_index(mean.index)]
    std_ = std[col_name][mask_index(std.index)]
    
    ax.plot(mean_, label=d.name)
    ax.fill_between(
        mean_.index, mean_ - std_, mean_ + std_,
        alpha=.25, linewidth=0
    )

ax.legend(title='Parameters:')
ax.set_title(col_name)
ax.set_xlabel(index)
plt.show()
#%%
