import os
import pandas as pd
import numpy as np
import torch
import warnings

warnings.simplefilter('ignore')

from run_avaed_ablations import MockArgs
from upstream import select_variables, build_site_split_indices, label_encode_dataframe, zscore, _vaed_train_and_cluster

args = MockArgs()
K = 3
args.vaed_epochs = 500  
args.vaed_clusters = K

df = pd.read_excel(args.data, sheet_name=args.sheet)

# 获取原本代码名称如 AKA，存储起来
real_station_names = df["Ecological Station Code"].copy()

group_candidates = ['Ecological Station Code', '生态站代码', 'station', 'Station']
for gc in group_candidates:
    if gc in df.columns:
        args.group_col = gc
        break

df[args.group_col] = df[args.group_col].astype('category').cat.codes.apply(lambda x: f"Station_{x}")

if df[args.time_col].dtype.kind not in "iu":
    df[args.time_col] = pd.to_numeric(df[args.time_col], errors='coerce')
df = df.dropna(subset=[args.time_col])
df[args.time_col] = df[args.time_col].astype(float)

exclude = [args.group_col]
var_cols = select_variables(df, args.target_col, exclude_cols=exclude)
target_idx = var_cols.index(args.target_col)

train_idx, val_idx = build_site_split_indices(pd, np, df, args.group_col, args.train_sites, args.test_sites, split_seed=args.split_seed)

tr_df_full = df.loc[train_idx]
tr_y_num = pd.to_numeric(tr_df_full[args.target_col], errors='coerce')
tr_mask_y = tr_y_num.notna() & (tr_y_num != 0)
train_idx_eff = tr_df_full[tr_mask_y].index.tolist()

work_tr_fit = tr_df_full[var_cols].copy()
encoders, _ = label_encode_dataframe(pd, work_tr_fit, exclude_numeric=[])
work_tr_full_enc = work_tr_fit
work_tr = work_tr_full_enc.loc[train_idx_eff]

Xtr = work_tr.to_numpy(dtype=float)
Xtr = np.where(np.isnan(Xtr), 0.0, Xtr)
Xtr[:, target_idx] = 0.0
Xtr, meanX, stdX = zscore(pd, np, Xtr)

station_labels = tr_df_full[args.group_col].loc[train_idx_eff].to_numpy()

real_station_labels = real_station_names.loc[train_idx_eff].to_numpy()

current_seed = args.split_seed + 100

vaed_res = _vaed_train_and_cluster(
    torch, np, Xtr,
    K=K, z_dim=args.vaed_latent_dim, hidden=args.vaed_hidden,
    epochs=args.vaed_epochs, lr=args.vaed_lr, lambda3=args.vaed_lambda3,
    gmm_update_every=args.vaed_gmm_update, seed=current_seed,
    station_labels=station_labels, outdir=args.outdir
)

os.makedirs(args.outdir, exist_ok=True)
pd.DataFrame(vaed_res['mu']).to_csv(os.path.join(args.outdir, f'avaed_K{K}_latent_mu.csv'), index=False)
pd.DataFrame(vaed_res['resp']).to_csv(os.path.join(args.outdir, f'avaed_K{K}_latent_resp.csv'), index=False)
pd.DataFrame(real_station_labels, columns=["Station"]).to_csv(os.path.join(args.outdir, f'avaed_K{K}_real_station_labels.csv'), index=False)

print(f"Extraction completed. Saved to {args.outdir}")
