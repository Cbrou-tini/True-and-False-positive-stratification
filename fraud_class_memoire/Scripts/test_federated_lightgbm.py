#!/usr/bin/env python
# coding: utf-8
#
# Federated LightGBM test script -- same "Fed XGBoost" architecture as the
# tail of H2_script.ipynb 

import time
import os
import re
import glob
import gc

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
from tqdm.auto import tqdm

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score, make_scorer,
                              average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.callbacks import DeadlineStopper


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sp(*parts):
    """Resolve a path relative to this script's own directory (Scripts/),
    regardless of the cwd the script happens to be launched from."""
    return os.path.join(SCRIPT_DIR, *parts)


IBAN_COL   = 'Account'
TS_COL     = 'Timestamp'
LABEL_COL  = 'proxy_label'
HOLDING_PSP_COL = 'To Bank'
DECLARING_PSP_COL = 'From Bank'

out = pd.read_csv(sp('..', 'dataset', 'fake_fncrf.csv'),
                   on_bad_lines='warn', engine='python')
out['bank'] = out['bank'].astype(str).str.strip()
out = out[out['bank'] != '70']

combined = out
combined = combined.drop(columns=['Is Laundering', 'nb.currency', 'delta.t', 'currency.mismatch',
       'is.self.transfer', 'is.intra.bank', 'log.amount', 'is.round.amount', 'hour.of.day',
       'day.of.week', 'is.off.hours', 'nb.distinct.to.bank_cum', 'nb.distinct.from.bank_cum',
       'nb.distinct.payfmt_cum', 'top.1.holder.RC', 'top.1.holder.SC', 'nb.iban.holder',
       'nb.events.holder', 'top.1.declaring.RC', 'top.1.declaring.SC', 'nb.iban.declaring',
       'nb.events.declaring', 'fan.out', 'fan.in', 'fan.ratio', 'key', 'y_pred'])
bank_col = 'bank'


def cumulative_nunique_vectorized(df, group_col, value_col):
    dup = df.duplicated(subset=[group_col, value_col])
    first_seen = (~dup).astype(int)
    return first_seen.groupby(df[group_col]).cumsum()


def get_item(bic, d, str_key):
    entry = d.get(str(bic) if not pd.isna(bic) else 'nan')
    return entry[str_key] if entry else float('nan')


def features_creator(dataset):
    dataset = dataset.sort_values(['Account', 'Timestamp']).reset_index(drop=True)
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

    acc_grp = dataset.groupby('Account')

    dataset['nb.currency'] = acc_grp.cumcount() + 1
    dataset['delta.t'] = acc_grp['Timestamp'].diff().dt.total_seconds().fillna(0)

    dataset['currency.mismatch'] = (dataset['Receiving Currency'] != dataset['Payment Currency']).astype(int)
    dataset['is.self.transfer']  = (dataset['Account'] == dataset['Account.1']).astype(int)
    dataset['is.intra.bank']     = (dataset['From Bank'] == dataset['To Bank']).astype(int)

    dataset['log.amount']      = np.log1p(dataset['Amount Paid'].astype(float))
    dataset['Amount Paid'] = pd.to_numeric(dataset['Amount Paid'], errors='coerce')
    dataset['is.round.amount'] = (dataset['Amount Paid'] % 100 == 0).astype(int)

    dataset['hour.of.day']  = dataset['Timestamp'].dt.hour
    dataset['day.of.week']  = dataset['Timestamp'].dt.dayofweek
    dataset['is.off.hours'] = dataset['Timestamp'].dt.hour.between(0, 5).astype(int)

    dataset['nb.distinct.to.bank_cum']   = cumulative_nunique_vectorized(dataset, 'Account', 'To Bank')
    dataset['nb.distinct.from.bank_cum'] = cumulative_nunique_vectorized(dataset, 'Account', 'From Bank')
    dataset['nb.distinct.payfmt_cum']    = cumulative_nunique_vectorized(dataset, 'Account', 'Payment Format')

    dict_bic_holding = {}
    for x, obj in dataset.groupby('To Bank'):
        vc_2 = obj['Receiving Currency'].value_counts()
        vc_3 = obj['Payment Currency'].value_counts()
        dict_bic_holding[str(x)] = {
            'top_RC': vc_2.index[0],
            'top_SC': vc_3.index[0],
            'nb.events.holding': obj['Timestamp'].nunique(),
            'nb.iban.holding': obj['Account'].nunique()
        }

    dict_bic_declaring = {}
    for x, obj in dataset.fillna({'From Bank': 'UNKNW'}).groupby('From Bank'):
        vc_2 = obj['Receiving Currency'].value_counts()
        vc_3 = obj['Payment Currency'].value_counts()
        dict_bic_declaring[str(x)] = {
            'top_RC': vc_2.index[0],
            'top_SC': vc_3.index[0],
            'nb.events.declaring': obj['Timestamp'].nunique(),
            'nb.iban.declaring': obj['Account'].nunique()
        }

    fan_out = dataset.groupby('Account')['Account.1'].nunique().rename('fan.out')
    fan_in  = dataset.groupby('Account.1')['Account'].nunique().rename('fan.in')

    to_bank_str   = dataset['To Bank'].apply(lambda b: str(b) if not pd.isna(b) else 'nan')
    from_bank_str = dataset['From Bank'].apply(lambda b: str(b) if not pd.isna(b) else 'nan')

    dataset['top.1.holder.RC']  = to_bank_str.map(lambda k: get_item(k, dict_bic_holding, 'top_RC'))
    dataset['top.1.holder.SC']  = to_bank_str.map(lambda k: get_item(k, dict_bic_holding, 'top_SC'))
    dataset['nb.iban.holder']   = to_bank_str.map(lambda k: get_item(k, dict_bic_holding, 'nb.iban.holding'))
    dataset['nb.events.holder'] = to_bank_str.map(lambda k: get_item(k, dict_bic_holding, 'nb.events.holding'))

    dataset['top.1.declaring.RC']  = from_bank_str.map(lambda k: get_item(k, dict_bic_declaring, 'top_RC'))
    dataset['top.1.declaring.SC']  = from_bank_str.map(lambda k: get_item(k, dict_bic_declaring, 'top_SC'))
    dataset['nb.iban.declaring']   = from_bank_str.map(lambda k: get_item(k, dict_bic_declaring, 'nb.iban.declaring'))
    dataset['nb.events.declaring'] = from_bank_str.map(lambda k: get_item(k, dict_bic_declaring, 'nb.events.declaring'))

    del dict_bic_declaring, dict_bic_holding

    dataset['fan.out']   = dataset['Account'].map(fan_out).fillna(0)
    dataset['fan.in']    = dataset['Account'].map(fan_in).fillna(0)
    dataset['fan.ratio'] = dataset['fan.in'] / (dataset['fan.out'] + 1)

    dataset['key'] = dataset['Account']

    return dataset


combined = combined.sort_values(TS_COL).reset_index(drop=True)
combined[TS_COL] = pd.to_datetime(combined[TS_COL], format='mixed', errors='coerce')
combined = combined.dropna(subset=[TS_COL])
combined[LABEL_COL] = combined[LABEL_COL].map({'Fraudeur': 1, 'Faux Positif': 0}).astype(float)

n_splits = 5
t_min, t_max = combined[TS_COL].min(), combined[TS_COL].max()
edges = pd.date_range(t_min, t_max, periods=n_splits + 2)
gap = pd.Timedelta(hours=1)

train_end = edges[1]
test_start = train_end + gap
test_end = edges[2]
train_df = combined[combined[TS_COL] < train_end].assign(fold=0)
test_df  = combined[(combined[TS_COL] >= test_start) & (combined[TS_COL] < test_end)].assign(fold=0)

del combined, out
gc.collect()

y_train = train_df[LABEL_COL]
y_test = test_df[LABEL_COL]

print("banks:", train_df[bank_col].nunique())

exclude = ['Is Laundering', 'Account', 'Account.1', 'Timestamp', 'y_pred', 'bank',
           'Amount Received', 'Amount Paid', 'key', LABEL_COL]

full_feats = features_creator(train_df.copy())
full_feats['delta.t'] = np.log1p(full_feats['delta.t'])

cat_cols = [c for c in full_feats.select_dtypes(include=['object', 'category']).columns
            if c not in exclude]
num_cols = [c for c in full_feats.select_dtypes(include=['number']).columns
            if c not in exclude]

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
])
cat_pipe.fit(full_feats[cat_cols])

num_imputer = SimpleImputer(strategy='median')
num_imputer.fit(full_feats[num_cols])
num_scaler = StandardScaler().fit(num_imputer.transform(full_feats[num_cols]))

del full_feats
gc.collect()

os.makedirs(sp('client_data'), exist_ok=True)

expected_width = len(num_cols) + len(cat_cols)
for f in glob.glob(sp('client_data', '*.pt')):
    try:
        x, _ = torch.load(f)
        if x.shape[1] != expected_width:
            raise ValueError(f"stale width {x.shape[1]} != expected {expected_width}")
    except Exception as e:
        print('corrupt/stale client_data, removing:', f, '-', e)
        os.remove(f)

LABEL_TMP_COL = '__y_train__'
needed_bids = {
    str(bid) for bid, group in train_df.groupby(bank_col) if len(group) >= 5
} - {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(sp('client_data_meta_lgbm', '*.pt'))}

print(f"rebuilding client_data/ for {len(needed_bids)} clients still missing cached margin features "
      f"(clients with cached margins skip this entirely)")

for bid, group in tqdm(train_df.groupby(bank_col), desc="Rebuilding client_data", unit="client"):
    if str(bid) not in needed_bids:
        continue

    out_path = sp('client_data', f'{bid}.pt')
    if os.path.exists(out_path):
        continue  # is already built

    group = group.copy()
    group[LABEL_TMP_COL] = y_train.loc[group.index].values

    group = features_creator(group)
    group['delta.t'] = np.log1p(group['delta.t'])

    cat_x = cat_pipe.transform(group[cat_cols]).astype(np.float32)
    num_x = num_scaler.transform(num_imputer.transform(group[num_cols])).astype(np.float32)

    x = torch.tensor(np.hstack([num_x, cat_x]), dtype=torch.float32)
    y = torch.tensor(group[LABEL_TMP_COL].values, dtype=torch.long)

    tmp_path = out_path + '.tmp'
    torch.save((x, y), tmp_path)
    os.replace(tmp_path, out_path)
    del x, y, cat_x, num_x, group

gc.collect()

# ---- held-out evaluation set ----
test_group = test_df.copy()
test_group['__y_test__'] = y_test.loc[test_group.index].values
test_group = features_creator(test_group)
test_group['delta.t'] = np.log1p(test_group['delta.t'])

cat_x_eval = cat_pipe.transform(test_group[cat_cols]).astype(np.float32)
num_x_eval = num_scaler.transform(num_imputer.transform(test_group[num_cols])).astype(np.float32)

x_eval = torch.tensor(np.hstack([num_x_eval, cat_x_eval]), dtype=torch.float32)
y_eval = torch.tensor(test_group['__y_test__'].values, dtype=torch.long)

del test_group, cat_x_eval, num_x_eval
gc.collect()

def _client_ids(pattern):
    return {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(sp(*pattern))}


client_data_ids = _client_ids(('client_data', '*.pt'))
cached_booster_ids = _client_ids(('client_lgbm_boosters', '*.txt'))
client_ids = sorted(client_data_ids | cached_booster_ids)

print(f"{len(client_ids)} client ids found "
      f"({len(client_data_ids)} with client_data/ tensors, "
      f"{len(cached_booster_ids)} with cached tuned boosters)")
if not client_ids:
    raise SystemExit(f"No client_data/*.pt under {sp('client_data')} and no cached boosters "
                      f"under {sp('client_lgbm_boosters')} -- run the client-partitioning "
                      "cells in H2_script.ipynb first (this script reuses those tensors).")


# 2. FedAdam infrastructure 

CLASS_WEIGHTS = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
CLASS_WEIGHTS_T = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS_T)


def client_update(model, x, y, epochs=3, lr=0.01):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    return model.state_dict()


class FedAdamServer:
    def __init__(self, global_state, lr=5e-4, beta1=0.9, beta2=0.99, eps=1e-3, clip_norm=1.0):
        self.state = {k: v.clone() for k, v in global_state.items()}
        self.m = {k: torch.zeros_like(v) for k, v in global_state.items()}
        self.v = {k: torch.zeros_like(v) for k, v in global_state.items()}
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.C = clip_norm

    def _clip_client_delta(self, client_delta):
        total_norm = torch.sqrt(sum((v ** 2).sum() for v in client_delta.values()))
        clip_coef = min(1.0, self.C / (total_norm + 1e-6))
        return {k: v * clip_coef for k, v in client_delta.items()}

    def step(self, client_states, client_sizes):
        L = len(client_states)
        clipped_deltas = []
        for cs in client_states:
            delta = {k: cs[k] - self.state[k] for k in self.state}
            clipped_deltas.append(self._clip_client_delta(delta))

        avg_delta = {k: sum(cd[k] for cd in clipped_deltas) / L for k in self.state}

        self.t += 1
        for k in self.state:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * avg_delta[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (avg_delta[k] ** 2)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            self.state[k] += self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return self.state


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    model.train()
    return loss, acc


def torch_predict_proba(model, x):
    model.eval()
    with torch.no_grad():
        proba = F.softmax(model(x), dim=1).numpy()
    model.train()
    return proba


def report_torch_model(name, model, x, y_true_tensor):
    proba = torch_predict_proba(model, x)
    preds = proba.argmax(axis=1)
    y_true = y_true_tensor.numpy()
    print(f"\n=== {name} ===")
    print(classification_report(y_true, preds))
    print(f"ROC-AUC: {roc_auc_score(y_true, proba[:, 1]):.4f}")
    return preds, proba


# 3. Per-client LightGBM tuning via Bayesian optimisation.

BO_TIME_BUDGET_S = 45  # per-client wall-clock cap for the BO search
FIXED_N_ESTIMATORS = 3  # number of boosting rounds
FALLBACK_PARAMS = dict(num_leaves=31, max_depth=6, learning_rate=0.1,
                        n_estimators=FIXED_N_ESTIMATORS, min_child_samples=10)

LGBM_SEARCH_SPACES = {
    'num_leaves': Integer(7, 255),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(1e-2, 0.3, prior='log-uniform'),
    'min_child_samples': Integer(5, 50),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'reg_alpha': Real(1e-6, 10.0, prior='log-uniform'),
    'reg_lambda': Real(1e-6, 10.0, prior='log-uniform'),
}

_bo_scoring = make_scorer(average_precision_score, response_method="predict_proba")


def adaptive_bo_iters(n_rows): #we are compute starved
    if n_rows < 300:
        return 10
    elif n_rows < 1000:
        return 8
    elif n_rows < 5000:
        return 6
    elif n_rows < 20000:
        return 4
    else:
        return 3


def train_local_lgbm_tuned(x, y):
    sample_weight = compute_sample_weight(class_weight='balanced', y=y)
    class_counts = np.bincount(y.astype(int), minlength=2)
    min_class = class_counts.min()

    if min_class < 2 or len(y) < 20:
        model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1,
                                    verbosity=-1, **FALLBACK_PARAMS)
        model.fit(x, y, sample_weight=sample_weight)
        return model.booster_, False

    n_splits = min(3, int(min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_iter = adaptive_bo_iters(len(y))
    deadline = DeadlineStopper(total_time=BO_TIME_BUDGET_S)

    model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1, verbosity=-1,
                                n_estimators=FIXED_N_ESTIMATORS)
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=LGBM_SEARCH_SPACES,
        scoring=_bo_scoring,
        n_iter=n_iter,
        cv=cv,
        n_jobs=1,
        verbose=0,
        random_state=42,
        refit=True,
    )
    try:
        opt.fit(x, y, sample_weight=sample_weight, callback=deadline)
        return opt.best_estimator_.booster_, True
    except Exception as e:
        print(f"  BO failed ({e!r}), falling back to fixed params")
        model = lgb.LGBMClassifier(objective='binary', random_state=42, n_jobs=-1,
                                    verbosity=-1, **FALLBACK_PARAMS)
        model.fit(x, y, sample_weight=sample_weight)
        return model.booster_, False


# 4. Merging independently-tuned LightGBM boosters into one frozen "global" booster

_TREE_SPLIT_RE = re.compile(r'\n\n(?=Tree=)')


def _split_model_text(model_str):
    header, rest = model_str.split('Tree=0', 1)
    body, footer = rest.split('end of trees', 1)
    body = 'Tree=0' + body
    blocks = _TREE_SPLIT_RE.split(body.strip('\n'))
    return header, blocks, footer


def merge_lgbm_boosters(boosters):
    # same logic as xgboost
    header0, footer0 = None, None
    all_blocks = []
    for b in boosters:
        header, blocks, footer = _split_model_text(b.model_to_string())
        if header0 is None:
            header0, footer0 = header, footer
        all_blocks.extend(blocks)

    renumbered = [re.sub(r'^Tree=\d+', f'Tree={i}', blk) for i, blk in enumerate(all_blocks)]
    new_body = '\n\n'.join(renumbered)
    tree_sizes = [len(blk.encode('utf-8')) + 2 for blk in renumbered]
    header = re.sub(r'tree_sizes=.*', 'tree_sizes=' + ' '.join(map(str, tree_sizes)), header0)

    merged_str = header + new_body + '\n\nend of trees' + footer0
    merged = lgb.Booster(model_str=merged_str)
    return merged, len(all_blocks)


def _leaf_value_array(tree_structure):
    values = {}
    stack = [tree_structure]
    while stack:
        node = stack.pop()
        if 'left_child' in node:
            stack.append(node['left_child'])
            stack.append(node['right_child'])
        else:
            values[node.get('leaf_index', 0)] = node['leaf_value']
    arr = np.zeros(max(values) + 1, dtype=np.float32)
    for k, v in values.items():
        arr[k] = v
    return arr


def build_leaf_arrays(booster):
    dump = booster.dump_model()
    return [_leaf_value_array(t['tree_structure']) for t in dump['tree_info']]


def get_lgbm_tree_margins(booster, x, total_trees, leaf_arrays):
    leaf_idx = booster.predict(x, pred_leaf=True)
    if leaf_idx.ndim == 1:
        leaf_idx = leaf_idx.reshape(-1, 1)

    N = x.shape[0]
    margins = np.zeros((N, total_trees), dtype=np.float32)
    for t in range(total_trees):
        margins[:, t] = leaf_arrays[t][leaf_idx[:, t].astype(np.int64)]
    return margins


# 5. Phase 1: per-client BO-tuned LightGBM boosters -> merge into one frozen global booster

os.makedirs(sp('client_lgbm_boosters'), exist_ok=True)

local_boosters = []
tuned_count = 0
t_phase1 = time.time()

print("\n=== Federated LightGBM: Phase 1 (per-client BO tuning + tree bagging) ===")
pbar = tqdm(client_ids, desc="Phase 1: tuning clients", unit="client")
for bid in pbar:
    booster_path = sp('client_lgbm_boosters', f'{bid}.txt')

    if os.path.exists(booster_path):
        # already tuned in a previous run -- doesn't need client_data/ at all.
        booster = lgb.Booster(model_file=booster_path)
    else:
        data_path = sp('client_data', f'{bid}.pt')
        if not os.path.exists(data_path):
            raise SystemExit(
                f"Client {bid} has no cached booster ({booster_path}) and no "
                f"client_data tensor ({data_path}) -- client_data/ is only needed for "
                "clients that haven't been BO-tuned yet; restore it (or this client's "
                ".pt file) to tune the remaining clients."
            )
        x, y = torch.load(data_path)
        booster, tuned = train_local_lgbm_tuned(x.numpy(), y.numpy())
        tuned_count += int(tuned)
        tmp_path = booster_path + '.tmp'
        booster.save_model(tmp_path)
        os.replace(tmp_path, booster_path)
        del x, y

    local_boosters.append(booster)
    pbar.set_postfix(tuned=tuned_count, refresh=False)

print(f"Phase 1 done in {time.time() - t_phase1:.0f}s: {len(local_boosters)} local boosters, "
      f"{tuned_count} BO-tuned (rest fell back to fixed params for degenerate clients)")

global_booster, total_trees = merge_lgbm_boosters(local_boosters)
in_dim = total_trees
print(f"merged booster: {total_trees} trees total")

del local_boosters
gc.collect()

joblib.dump({
    "global_booster": global_booster,
    "total_trees": total_trees,
    "in_dim": in_dim,
}, sp("federated_lightgbm_margins.joblib"))

# dump_model() re-serializes the whole (now merged) booster -- do this ONCE
# here, not once per client inside the loop below (see build_leaf_arrays).
global_leaf_arrays = build_leaf_arrays(global_booster)


# =====================================================================
# 6. Phase 2: frozen-booster tree-margin features -> FedAdam-trained linear meta-learner
# =====================================================================

os.makedirs(sp('client_data_meta_lgbm'), exist_ok=True)

for f in glob.glob(sp('client_data_meta_lgbm', '*.pt')):
    try:
        feats, _ = torch.load(f)
        if feats.shape[1] != total_trees:
            raise ValueError(f"stale width {feats.shape[1]} != expected {total_trees}")
    except Exception as e:
        print('stale/corrupt margin cache, removing:', f, '-', e)
        os.remove(f) #corruptions issue happened 

skipped_no_data = []
for bid in tqdm(client_ids, desc="Phase 2: caching margin features", unit="client"):
    out_path = sp('client_data_meta_lgbm', f'{bid}.pt')
    if os.path.exists(out_path):
        continue
    data_path = sp('client_data', f'{bid}.pt')
    if not os.path.exists(data_path):
        skipped_no_data.append(bid)
        continue
    x, y = torch.load(data_path)
    margins = get_lgbm_tree_margins(global_booster, x.numpy(), total_trees, global_leaf_arrays)
    feats = torch.tensor(margins, dtype=torch.float32)
    tmp_path = out_path + '.tmp'
    torch.save((feats, y), tmp_path)
    os.replace(tmp_path, out_path)
    del x, y, margins, feats
gc.collect()

if skipped_no_data:
    print(f"WARNING: skipped {len(skipped_no_data)}/{len(client_ids)} clients with no "
          f"client_data/ tensor and no cached margin features -- they will NOT be part of "
          f"the meta-learner training below. Restore client_data/ for these clients and "
          f"rerun to include them: {skipped_no_data[:10]}"
          f"{' ...' if len(skipped_no_data) > 10 else ''}")

x_eval_margins = get_lgbm_tree_margins(global_booster, x_eval.numpy(), total_trees, global_leaf_arrays)
x_eval_meta = torch.tensor(x_eval_margins, dtype=torch.float32)
del x_eval_margins
gc.collect()

meta_files = sorted(glob.glob(sp('client_data_meta_lgbm', '*.pt')))
print(f"Phase 2 meta-learner will train on {len(meta_files)}/{len(client_ids)} clients "
      f"(those with cached margin features)")
meta_sizes = [torch.load(f)[1].shape[0] for f in meta_files]

print("\n=== Federated LightGBM: FedAdam + MLP (meta-learner) ===")

def make_meta_model():
    return nn.Linear(in_dim, 2)


meta_model = make_meta_model()
server = FedAdamServer(meta_model.state_dict(), lr=1e-3, clip_norm=1.0)

n_rounds = 100
round_pbar = tqdm(range(n_rounds), desc="Phase 2: FedAdam rounds", unit="round")
for rnd in round_pbar:
    client_states, sizes = [], []
    for f, n in tqdm(list(zip(meta_files, meta_sizes)), desc=f"round {rnd} clients",
                      unit="client", leave=False):
        x, y = torch.load(f)
        m = make_meta_model()
        m.load_state_dict(server.state)
        client_states.append(client_update(m, x, y))
        sizes.append(n)
        del x, y
    server.step(client_states, sizes)
    meta_model.load_state_dict(server.state)
    loss, acc = evaluate(meta_model, x_eval_meta, y_eval)
    round_pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}", refresh=True)

os.makedirs(sp("models_H2"), exist_ok=True)
torch.save(meta_model.state_dict(), sp("models_H2", "lgbm_meta_learner.pt"))
joblib.dump(
    {"global_booster": global_booster, "total_trees": total_trees, "in_dim": in_dim},
    sp("models_H2", "lgbm_global_booster.joblib"),
)

report_torch_model("Federated LightGBM meta-learner", meta_model, x_eval_meta, y_eval)


# ---- reload sanity check ----
loaded = joblib.load(sp("models_H2", "lgbm_global_booster.joblib"))
loaded_booster = loaded["global_booster"]
loaded_total_trees = loaded["total_trees"]
loaded_in_dim = loaded["in_dim"]

loaded_meta_model = make_meta_model(loaded_in_dim)
loaded_meta_model.load_state_dict(torch.load(sp("models_H2", "lgbm_meta_learner.pt")))
loaded_meta_model.eval()

loaded_leaf_arrays = build_leaf_arrays(loaded_booster)
x_eval_margins_check = get_lgbm_tree_margins(loaded_booster, x_eval.numpy(), loaded_total_trees, loaded_leaf_arrays)
x_eval_meta_check = torch.tensor(x_eval_margins_check, dtype=torch.float32)
del x_eval_margins_check

report_torch_model("Federated LightGBM meta-learner (loaded from disk)",
                    loaded_meta_model, x_eval_meta_check, y_eval)
