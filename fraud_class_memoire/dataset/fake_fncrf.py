
### packages
import pandas as pd
import sqlalchemy as sqla
import numpy as np
import re
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import glob
import os
import networkx as nx
from collections import defaultdict


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
    dataset['is.round.amount'] = (dataset['Amount Paid'] % 100 == 0).astype(int)

    dataset['hour.of.day']  = dataset['Timestamp'].dt.hour
    dataset['day.of.week']  = dataset['Timestamp'].dt.dayofweek
    dataset['is.off.hours'] = dataset['Timestamp'].dt.hour.between(0, 5).astype(int)

    dataset['nb.distinct.to.bank_cum']   = cumulative_nunique_vectorized(dataset, 'Account', 'To Bank')
    dataset['nb.distinct.from.bank_cum'] = cumulative_nunique_vectorized(dataset, 'Account', 'From Bank')
    dataset['nb.distinct.payfmt_cum']    = cumulative_nunique_vectorized(dataset, 'Account', 'Payment Format')

    # --- Holding PSP (keyed by To Bank) ---
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

    # --- Declaring PSP (keyed by From Bank) ---
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

    dataset['key'] = dataset['Account']  # replicates original df.assign(key=key) at concat time

    return dataset

path = r"./fraud_class_memoire/dataset/HI-Large_Trans.csv"
chunksize = 500_000
first_write = True
count = 0 
try: 
        for f in glob.glob("./fraud_class_memoire/datasetoutput_*.csv"):
            dataset = pd.read_csv(f)
            val = os.path.basename(f).replace('datasetoutput_', '').replace('.csv', '')
            dataset["Is Laundering"].value_counts()
            
            combined = dataset
            IBAN_COL   = 'Account'
            TS_COL     = 'Timestamp'
            LABEL_COL  = 'Is Laundering' 
            HOLDING_PSP_COL = 'To Bank'
            DECLARING_PSP_COL = 'From Bank'
            N          = 1 # maximum number o events taken into account 

            combined[TS_COL]    = pd.to_datetime(combined[TS_COL])

            SPLIT_DATE = combined[TS_COL].quantile(0.8) #we keep 20% of the data as a validation split
            train_df = combined[combined[TS_COL] <  SPLIT_DATE]
            test_df  = combined[combined[TS_COL] >= SPLIT_DATE]
            print(f"LEN TRAIN: {len(train_df)}, LEN TEST: {len(test_df)}")
            
            train_df = features_creator(train_df)
            test_df = features_creator(test_df)

            test_df['delta.t'] = np.log1p(test_df['delta.t'])
            train_df['delta.t'] = np.log1p(train_df['delta.t'])

            drop_cols = [LABEL_COL, IBAN_COL, TS_COL]
            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[LABEL_COL].values.astype('int64')

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[LABEL_COL].values.astype('int64')

            cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = train_df.select_dtypes(include=['number']).columns.tolist()

            # drop target/id/leakage columns from both
            exclude = ['Is Laundering', 'Account', 'Account.1', 'Timestamp', 'y_pred', 'bank', 'Amount Received', 'Amount Paid']
            cat_cols = [c for c in cat_cols if c not in exclude]
            num_cols = [c for c in num_cols if c not in exclude]

            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('cat', cat_pipe, cat_cols),
                ('num', num_pipe, num_cols)
            ])

            X_train = preprocessor.fit_transform(train_df[cat_cols + num_cols])
            X_test  = preprocessor.transform(test_df[cat_cols + num_cols])

            model = LogisticRegression(class_weight='balanced', max_iter=1000)
            model.fit(X_train, y_train)
            from sklearn.metrics import precision_recall_curve
            proba_train = model.predict_proba(X_train)[:, 1]
            proba_test  = model.predict_proba(X_test)[:, 1]

            prec, rec, thresh = precision_recall_curve(y_train, proba_train)
            TARGET_PRECISION = 0.08
            idx = next((i for i, p in enumerate(prec) if p >= TARGET_PRECISION), None)
            if idx is None or idx >= len(thresh):
                THRESH = 0.15 # <-- explicit high fallback, not the accidental lowest-threshold one
            else:
                THRESH = thresh[idx]

            preds = (proba_test >= THRESH).astype(int)

            test_prec, test_rec, _ = precision_recall_curve(y_test, proba_test)
            print(f"Threshold: {THRESH:.4f}")
            print(f"Train precision target: {TARGET_PRECISION}")
            print(f"Actual test precision at this threshold: "
                f"{(y_test[preds==1]==1).mean() if preds.sum()>0 else float('nan'):.4f}")
            print(f"Raw FP count: {((y_test==0)&(preds==1)).sum()}, Raw TP count: {((y_test==1)&(preds==1)).sum()}")

            print(classification_report(y_test, preds))
            print(f"ROC-AUC: {roc_auc_score(y_test, proba_test):.4f}")
            from sklearn.metrics import confusion_matrix
            print(confusion_matrix(y_test, preds))

            test_df = test_df.copy()
            test_df['y_pred'] = preds                                                    # <-- FIX: use tuned preds, not model.predict() default
            test_df['bank'] = val

            tp_idx = test_df[test_df['Is Laundering'] == 1].index
            fp_idx = test_df[(test_df['Is Laundering'] == 0) & (test_df['y_pred'] == 1)].index

            tp_rows = test_df.loc[tp_idx].copy()
            fp_rows = test_df.loc[fp_idx].copy()

            tp_rows = tp_rows.assign(proxy_label='Fraudeur')
            fp_rows = fp_rows.assign(proxy_label='Faux Positif')
            tp_rows['bank'] = val
            fp_rows['bank'] = val

            out = pd.concat([tp_rows, fp_rows])
            print(out.head())
            out.to_csv('./dataset/fake_fncrf.csv', mode='a', header=first_write, index=True)
            first_write = False
except Exception as e:
        print(e)

