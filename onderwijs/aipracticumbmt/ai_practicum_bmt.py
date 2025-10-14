# %% ai_practicum_bmt.py
# Compare Linear Regression and Random Forest for cholesterol prediction

import pandas as pd
import numpy as np

# 1) Load data
url = 'http://wptmdoorn.github.io/data/FTIR%20data_train_studenten.csv'
T = pd.read_csv(url, delimiter=';', engine='python')

labelVar = 'cholesterol_mmol_L'

if labelVar not in T.columns:
    raise ValueError(f'Label variable "{labelVar}" not found in the dataset.')

# 2) Prepare predictors (numeric only)
Xtable = T.select_dtypes(include=[np.number]).copy()

if labelVar in Xtable.columns:
    Xtable = Xtable.drop(columns=[labelVar])

y = T[labelVar]

# Remove rows with missing values
valid = (~y.isna()) & (~Xtable.isna().any(axis=1))
Xtable = Xtable.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

X = Xtable.to_numpy()
featureNames = Xtable.columns.tolist()

# 3) Train/test split — classic 80/20 (manual)
np.random.seed(0)
n = X.shape[0]
idx = np.random.permutation(n)
ntr = int(round(0.8 * n))

idxTr = idx[:ntr]
idxTe = idx[ntr:]

Xtr, Xte = X[idxTr, :], X[idxTe, :]
ytr, yte = y.iloc[idxTr], y.iloc[idxTe]

# 3b) Print dimensions
print(f'Total samples: {n} | Features: {X.shape[1]}')
print(f'Train set: {Xtr.shape[0]} x {Xtr.shape[1]} (Xtr) | {len(ytr)} (ytr)')
print(f'Test  set: {Xte.shape[0]} x {Xte.shape[1]} (Xte) | {len(yte)} (yte)')

# 3c) Print heads of tables
print('\n--- Head of original table T ---')
print(T.head())

print('\n--- Head of numeric predictor table Xtable ---')
print(Xtable.head())

print('\n--- Head of training predictors (as DataFrame) ---')
print(pd.DataFrame(Xtr, columns=featureNames).head())

print('\n--- Head of test predictors (as DataFrame) ---')
print(pd.DataFrame(Xte, columns=featureNames).head())

print('\n--- Head of training response ytr ---')
print(ytr.head())

print('\n--- Head of test response yte ---')
print(yte.head())
