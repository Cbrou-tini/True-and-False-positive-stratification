"""
Correlation plot of the "local" features that were computed during database
creation (dataset/fake_fncrf.py: features_creator()) and are later dropped
in Feature_Extraction.ipynb (cell 8) before the notebook's own
features_creator() rebuilds a fresh set of event/lag features.
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = r'/Users/cbrou/Documents/fraud_class_memoire/dataset/fake_fncrf.csv'
OUT_PATH = r'/Users/cbrou/Documents/fraud_class_memoire/correlation_local_features.png'

LOCAL_FEATURES = [
    'nb.currency', 'delta.t', 'currency.mismatch', 'is.self.transfer',
    'is.intra.bank', 'log.amount', 'is.round.amount', 'hour.of.day',
    'day.of.week', 'is.off.hours', 'nb.distinct.to.bank_cum',
    'nb.distinct.from.bank_cum', 'nb.distinct.payfmt_cum',
    'top.1.holder.RC', 'top.1.holder.SC', 'nb.iban.holder',
    'nb.events.holder', 'top.1.declaring.RC', 'top.1.declaring.SC',
    'nb.iban.declaring', 'nb.events.declaring', 'fan.out', 'fan.in',
    'fan.ratio',
]

out = pd.read_csv(DATA_PATH, on_bad_lines='warn', low_memory=False)
out['bank'] = out['bank'].astype(str).str.strip()
out = out[out['bank'] != '70']

local_df = out[LOCAL_FEATURES].copy()
for col in local_df.columns:
    local_df[col] = pd.to_numeric(local_df[col], errors='coerce')

# drop columns that ended up fully non-numeric / constant (can't correlate)
local_df = local_df.loc[:, local_df.nunique(dropna=True) > 1]

corr = local_df.corr()

fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)

fig.colorbar(im, ax=ax, label='Pearson correlation')
ax.set_title('Correlation of local features')
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f'Saved to {OUT_PATH}')
print(local_df.columns.tolist())
