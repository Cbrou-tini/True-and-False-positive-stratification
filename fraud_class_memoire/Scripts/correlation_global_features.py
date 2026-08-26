
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = r'./Users/cbrou/Documents/fraud_class_memoire/dataset/fake_fncrf.csv'
OUT_PATH = r'./fraud_class_memoire/correlation_global_features.png'
LABEL_COL = 'proxy_label'

out = pd.read_csv(DATA_PATH, on_bad_lines='warn', low_memory=False)
out['bank'] = out['bank'].astype(str).str.strip()
out = out[out['bank'] != '70'] #Excluded due too size

dataset = out.sort_values(['Account', 'Timestamp']).reset_index(drop=True)
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], format='mixed', errors='coerce')
dataset = dataset.dropna(subset=['Timestamp'])

# --- Cross-bank features ---
for bank_col, prefix in [('From Bank', 'declaring'), ('To Bank', 'holding')]:
    bank_key = dataset[bank_col].fillna('UNKNW')
    grp = dataset.groupby(bank_key)

    is_fraud = (dataset[LABEL_COL] == 'Fraudeur').astype(int)
    is_fp    = (dataset[LABEL_COL] == 'Faux Positif').astype(int)

    prior_fraud_cnt = is_fraud.groupby(bank_key).cumsum().groupby(bank_key).shift(1)
    prior_fp_cnt    = is_fp.groupby(bank_key).cumsum().groupby(bank_key).shift(1)
    prior_n         = grp.cumcount()

    dataset[f'{prefix}.fraud_rate']   = (prior_fraud_cnt / prior_n.replace(0, np.nan)).fillna(0)
    dataset[f'{prefix}.fp_rate']      = (prior_fp_cnt / prior_n.replace(0, np.nan)).fillna(0)
    dataset[f'{prefix}.nb.prior.txn'] = prior_n
    dataset[f'{prefix}.has.history']  = (prior_n > 0).astype(int)

# --- Corridor fraud rate ---
corridor_key = dataset['From Bank'].fillna('UNKNW').astype(str) + '_' + dataset['To Bank'].fillna('UNKNW').astype(str)
is_fraud = (dataset[LABEL_COL] == 'Fraudeur').astype(int)
corridor_prior_fraud = is_fraud.groupby(corridor_key).cumsum().groupby(corridor_key).shift(1)
corridor_prior_n     = dataset.groupby(corridor_key).cumcount()
dataset['corridor.fraud_rate']   = (corridor_prior_fraud / corridor_prior_n.replace(0, np.nan)).fillna(0)
dataset['corridor.nb.prior.txn'] = corridor_prior_n

GLOBAL_FEATURES = [
    'declaring.fraud_rate', 'declaring.fp_rate', 'declaring.nb.prior.txn', 'declaring.has.history',
    'holding.fraud_rate', 'holding.fp_rate', 'holding.nb.prior.txn', 'holding.has.history',
    'corridor.fraud_rate', 'corridor.nb.prior.txn',
]

global_df = dataset[GLOBAL_FEATURES].apply(pd.to_numeric, errors='coerce')
corr = global_df.corr()

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)

fig.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Correlation of global features')
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f'Saved to {OUT_PATH}')
print(global_df.columns.tolist())
