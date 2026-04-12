import argparse
import os
import sys
from typing import List, Dict, Tuple, Optional
from typing import Any
from upstream import *

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, target_idx: int, adj: 'torch.Tensor'):
        super().__init__()
        self.layers = layers
        self.target_idx = target_idx
        self.adj = adj
        N = in_dim
        self.W_self = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(N, N) * 0.05) for _ in range(layers)])
        self.W_neigh = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(N, N) * 0.05) for _ in range(layers)])
        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(N)) for _ in range(layers)])
        self.dropout = torch.nn.Dropout(p=0.1)
        self.out_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = x
        A = self.adj
        A_norm = A / (A.sum(dim=1, keepdim=True) + 1e-08)
        for i in range(self.layers):
            neigh = torch.matmul(h, A_norm)
            hs = torch.matmul(h, self.W_self[i])
            hn = torch.matmul(neigh, self.W_neigh[i])
            pre = hs + hn + self.bias[i]
            if i < self.layers - 1:
                h = torch.nn.functional.leaky_relu(pre, negative_slope=0.1)
                h = self.dropout(h)
            else:
                h = pre
        out = h[:, self.target_idx] + self.out_bias
        return out

class ECMPNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, target_idx: int, adj: 'torch.Tensor'):
        super().__init__()
        self.layers = layers
        self.target_idx = target_idx
        self.adj = adj
        N = in_dim
        self.W_self = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(N, N) * 0.05) for _ in range(layers)])
        self.W_neigh = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(N, N) * 0.05) for _ in range(layers)])
        self.edge_w = torch.nn.Parameter(torch.rand(N, N))
        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(N)) for _ in range(layers)])
        self.dropout = torch.nn.Dropout(p=0.1)
        self.out_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = x
        A = self.adj
        Ew = torch.sigmoid(self.edge_w) * A
        Ew_norm = Ew / (Ew.sum(dim=1, keepdim=True) + 1e-08)
        for i in range(self.layers):
            neigh = torch.matmul(h, Ew_norm)
            hs = torch.matmul(h, self.W_self[i])
            hn = torch.matmul(neigh, self.W_neigh[i])
            pre = hs + hn + self.bias[i]
            if i < self.layers - 1:
                h = torch.nn.functional.leaky_relu(pre, negative_slope=0.1)
                h = self.dropout(h)
            else:
                h = pre
        out = h[:, self.target_idx] + self.out_bias
        return out

class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, target_idx: int, adj: 'torch.Tensor'):
        super().__init__()
        self.layers = layers
        self.target_idx = target_idx
        N = in_dim
        A = adj
        A_sym = 0.5 * (A + A.T)
        deg = torch.clamp(A_sym.sum(dim=1), min=1e-08)
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        self.register_buffer('A_norm', D_inv_sqrt @ A_sym @ D_inv_sqrt)
        self.W = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(N, N) * 0.05) for _ in range(layers)])
        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(N)) for _ in range(layers)])
        self.dropout = torch.nn.Dropout(p=0.1)
        self.out_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = x
        A_norm = self.A_norm
        for i in range(self.layers):
            neigh = torch.matmul(h, A_norm)
            pre = torch.matmul(neigh, self.W[i]) + self.bias[i]
            if i < self.layers - 1:
                h = torch.nn.functional.leaky_relu(pre, negative_slope=0.1)
                h = self.dropout(h)
            else:
                h = pre
        out = h[:, self.target_idx] + self.out_bias
        return out

def predict_test_sites(torch, np, df_test, var_cols, target_col, time_col, group_col, model, target_idx, meanX, stdX, meany, stdy, encoders):
    predictions = []
    import pandas as pd
    work_test = apply_encoders(pd, df_test[var_cols].copy(), encoders)
    for site, group in df_test.groupby(group_col):
        group = group.sort_values(time_col)
        group_work = work_test.loc[group.index]
        X_test = group_work.to_numpy(dtype=float)
        y_true = X_test[:, target_idx].copy()
        X_test[:, target_idx] = 0.0
        X_test = (X_test - meanX) / stdX
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).cpu().numpy()
        pred = pred * (stdy + 1e-08) + meany
        for i, (idx, row) in enumerate(group.iterrows()):
            predictions.append({group_col: row[group_col], time_col: row[time_col], 'y_true': row[target_col], 'y_pred': pred[i]})
    return predictions

def train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, model_type: str, adj, hidden: int, epochs: int, lr: float, target_idx: int, outdir: str, tag: str):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cpu')
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    A_t = torch.tensor(adj, dtype=torch.float32, device=device)
    in_dim = Xtr_t.shape[1]
    layers = 2
    if model_type == 'GraphSAGE':
        model = GraphSAGE(in_dim, hidden, layers, target_idx, A_t)
    elif model_type == 'GCN':
        model = GCN(in_dim, hidden, layers, target_idx, A_t)
    else:
        model = ECMPNN(in_dim, hidden, layers, target_idx, A_t)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    loss_fn = torch.nn.MSELoss()
    best = {'val_mse': float('inf'), 'state': None}
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xva_t)
                mse = float(loss_fn(pv, yva_t).item())
            if mse < best['val_mse']:
                best['val_mse'] = mse
                best['state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        pred_tr = model(Xtr_t).cpu().numpy()
        pred_va = model(Xva_t).cpu().numpy()

    def metrics(y, p):
        mse = float(np.mean((y - p) ** 2))
        mae = float(np.mean(np.abs(y - p)))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-08
        r2 = 1 - ss_res / ss_tot
        return (mse, mae, r2)
    mtr = metrics(ytr, pred_tr)
    mva = metrics(yva, pred_va)
    import pandas as pd
    pd.DataFrame({'y_true': yva, 'y_pred': pred_va}).to_csv(os.path.join(outdir, f'pred_{model_type}_{tag}.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(outdir, f'metrics_{model_type}_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Train MSE={mtr[0]:.6f}, MAE={mtr[1]:.6f}, R2={mtr[2]:.4f}\n')
        f.write(f'Valid MSE={mva[0]:.6f}, MAE={mva[1]:.6f}, R2={mva[2]:.4f}\n')
    return (pred_tr, pred_va, mtr, mva)

def train_and_eval_rf(np, Xtr, ytr, Xva, yva, outdir: str, tag: str):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as e:
        raise RuntimeError('需要 scikit-learn 以运行随机森林：pip install scikit-learn')
    os.makedirs(outdir, exist_ok=True)
    rf = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    p_tr = rf.predict(Xtr)
    p_va = rf.predict(Xva)

    def metrics(y, p):
        mse = float(np.mean((y - p) ** 2))
        mae = float(np.mean(np.abs(y - p)))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-08
        r2 = 1 - ss_res / ss_tot
        return (mse, mae, r2)
    mtr = metrics(ytr, p_tr)
    mva = metrics(yva, p_va)
    import pandas as pd
    pd.DataFrame({'y_true': yva, 'y_pred': p_va}).to_csv(os.path.join(outdir, f'pred_RF_{tag}.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(outdir, f'metrics_RF_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Train MSE={mtr[0]:.6f}, MAE={mtr[1]:.6f}, R2={mtr[2]:.4f}\n')
        f.write(f'Valid MSE={mva[0]:.6f}, MAE={mva[1]:.6f}, R2={mva[2]:.4f}\n')
    return (p_tr, p_va, mtr, mva)

def train_and_eval_mlp(torch, np, Xtr, ytr, Xva, yva, hidden: int, epochs: int, lr: float, outdir: str, tag: str):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cpu')
    Xtr_s, mX, sX = ((Xtr - np.nanmean(Xtr, axis=0)) / (np.nanstd(Xtr, axis=0) + 1e-08), np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0) + 1e-08)
    Xva_s = (Xva - mX) / sX
    ytr_s, my, sy = ((ytr - np.nanmean(ytr)) / (np.nanstd(ytr) + 1e-08), np.nanmean(ytr), np.nanstd(ytr) + 1e-08)
    yva_s = (yva - my) / sy
    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr_s, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva_s, dtype=torch.float32, device=device)
    in_dim = Xtr_t.shape[1]
    model = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    best = {'val': float('inf'), 'state': None}
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t).squeeze(-1)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = model(Xva_t).squeeze(-1)
                mse = float(loss_fn(pv, yva_t).item())
            if mse < best['val']:
                best['val'] = mse
                best['state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        p_tr = model(Xtr_t).squeeze(-1).cpu().numpy()
        p_va = model(Xva_t).squeeze(-1).cpu().numpy()
    p_tr = p_tr * sy + my
    p_va = p_va * sy + my

    def metrics(y, p):
        mse = float(np.mean((y - p) ** 2))
        mae = float(np.mean(np.abs(y - p)))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-08
        r2 = 1 - ss_res / ss_tot
        return (mse, mae, r2)
    mtr = metrics(ytr, p_tr)
    mva = metrics(yva, p_va)
    import pandas as pd
    pd.DataFrame({'y_true': yva, 'y_pred': p_va}).to_csv(os.path.join(outdir, f'pred_MLP_{tag}.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(outdir, f'metrics_MLP_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Train MSE={mtr[0]:.6f}, MAE={mtr[1]:.6f}, R2={mtr[2]:.4f}\n')
        f.write(f'Valid MSE={mva[0]:.6f}, MAE={mva[1]:.6f}, R2={mva[2]:.4f}\n')
    return (p_tr, p_va, mtr, mva)

def train_and_eval_lgb(np, Xtr, ytr, Xva, yva, outdir: str, tag: str):
    """使用 LightGBM 回归进行训练与验证。"""
    os.makedirs(outdir, exist_ok=True)
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError(f'LightGBM 未安装: {e}')
    train_set = lgb.Dataset(Xtr, label=ytr)
    params = {'objective': 'regression', 'metric': ['l2', 'l1'], 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1, 'min_data_in_leaf': 20, 'verbose': -1}
    model = lgb.train(params, train_set, num_boost_round=1000)
    p_tr = model.predict(Xtr)
    p_va = model.predict(Xva)

    def metrics(y, p):
        mse = float(np.mean((y - p) ** 2))
        mae = float(np.mean(np.abs(y - p)))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-08
        r2 = 1 - ss_res / ss_tot
        return (mse, mae, r2)
    mtr = metrics(ytr, p_tr)
    mva = metrics(yva, p_va)
    import pandas as pd
    pd.DataFrame({'y_true': yva, 'y_pred': p_va}).to_csv(os.path.join(outdir, f'pred_LGB_{tag}.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(outdir, f'metrics_LGB_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Train MSE={mtr[0]:.6f}, MAE={mtr[1]:.6f}, R2={mtr[2]:.4f}\n')
        f.write(f'Valid MSE={mva[0]:.6f}, MAE={mva[1]:.6f}, R2={mva[2]:.4f}\n')
    return (p_tr, p_va, mtr, mva)

def train_and_eval_xgb(np, Xtr, ytr, Xva, yva, outdir: str, tag: str):
    """使用 XGBoost 回归进行训练与验证。"""
    os.makedirs(outdir, exist_ok=True)
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError(f'XGBoost 未安装: {e}')
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'eta': 0.05, 'max_depth': 6, 'subsample': 0.9, 'colsample_bytree': 0.9, 'lambda': 1.0, 'alpha': 0.0, 'verbosity': 0}
    model = xgb.train(params, dtr, num_boost_round=1000)
    p_tr = model.predict(dtr)
    p_va = model.predict(dva)

    def metrics(y, p):
        mse = float(np.mean((y - p) ** 2))
        mae = float(np.mean(np.abs(y - p)))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-08
        r2 = 1 - ss_res / ss_tot
        return (mse, mae, r2)
    mtr = metrics(ytr, p_tr)
    mva = metrics(yva, p_va)
    import pandas as pd
    pd.DataFrame({'y_true': yva, 'y_pred': p_va}).to_csv(os.path.join(outdir, f'pred_XGB_{tag}.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(outdir, f'metrics_XGB_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Train MSE={mtr[0]:.6f}, MAE={mtr[1]:.6f}, R2={mtr[2]:.4f}\n')
        f.write(f'Valid MSE={mva[0]:.6f}, MAE={mva[1]:.6f}, R2={mva[2]:.4f}\n')
    return (p_tr, p_va, mtr, mva)

def main(argv: List[str]) -> int:
    pd, np, nx, plt, torch = ensure_imports()
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    try:
        args.ivae_enabled = False
    except Exception:
        pass
    try:
        args.vaed_agg_threshold = min(float(getattr(args, 'vaed_agg_threshold', 0.2)), 0.1)
        args.avaed_agg_threshold = min(float(getattr(args, 'avaed_agg_threshold', args.vaed_agg_threshold)), 0.1)
        print(f'[设置] 已将 VAED 聚合阈值设为 {args.vaed_agg_threshold}, AVAED 聚合阈值设为 {args.avaed_agg_threshold}')
    except Exception:
        pass
    try:
        sheet_spec = args.sheet
        if isinstance(sheet_spec, str):
            s = sheet_spec.strip()
            if s.isdigit():
                sheet_spec = int(s)
        df = pd.read_excel(args.data, sheet_name=sheet_spec)
    except Exception as e:
        print(f'读取数据失败：{e}', file=sys.stderr)
        return 2
    if df[args.time_col].dtype.kind not in 'iu':
        df[args.time_col] = pd.to_numeric(df[args.time_col], errors='coerce')
    df = df.dropna(subset=[args.time_col])
    df[args.time_col] = df[args.time_col].astype(float)
    exclude = [args.group_col]
    var_cols = select_variables(df, args.target_col, exclude_cols=exclude)
    try:
        stats_df = compute_missing_zero_stats(pd, np, df, var_cols)
        stats_path = os.path.join(args.outdir, 'missing_zero_stats.csv')
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f'缺失/零值统计已保存：{stats_path}')
    except Exception as e:
        print(f'统计缺失/零值失败：{e}', file=sys.stderr)
    train_idx, val_idx = build_site_split_indices(pd, np, df, args.group_col, args.train_sites, args.test_sites, split_seed=int(getattr(args, 'split_seed', 42)))
    if len(train_idx) == 0 or len(val_idx) == 0:
        print('训练或验证集为空，请调整 train-sites 或 test-sites', file=sys.stderr)
        return 3
    tr_df_full = df.loc[train_idx]
    va_df_full = df.loc[val_idx]
    tr_y_num = pd.to_numeric(tr_df_full[args.target_col], errors='coerce')
    va_y_num = pd.to_numeric(va_df_full[args.target_col], errors='coerce')
    tr_mask_y = tr_y_num.notna() & (tr_y_num != 0)
    va_mask_y = va_y_num.notna() & (va_y_num != 0)
    train_idx_eff = tr_df_full[tr_mask_y].index.tolist()
    val_idx_eff = va_df_full[va_mask_y].index.tolist()
    work_tr_fit = tr_df_full[var_cols].copy()
    encoders, _ = label_encode_dataframe(pd, work_tr_fit, exclude_numeric=[])
    work_tr_full_enc = work_tr_fit
    work_va_full_enc = apply_encoders(pd, va_df_full[var_cols], encoders)
    work_tr = work_tr_full_enc.loc[train_idx_eff]
    work_va = work_va_full_enc.loc[val_idx_eff]
    try:
        fci_res = run_fci(np, work_tr, var_cols, args.alpha, os.path.join(args.outdir, args.graph_png), os.path.join(args.outdir, args.adj_out), debug_path=os.path.join(args.outdir, 'fci_debug.txt'))
    except Exception as e:
        print(f'FCI运行失败：{e}', file=sys.stderr)
        return 4
    if args.adjacency_mode == 'directed':
        adj_fci = fci_res['adj']
    elif args.adjacency_mode == 'semi':
        adj_fci = fci_res['semi']
    else:
        adj_fci = fci_res['skeleton']
    fs_mode = args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode
    if fs_mode == 'directed':
        adj_fci_fs = fci_res['adj']
    elif fs_mode == 'semi':
        adj_fci_fs = fci_res['semi']
    else:
        adj_fci_fs = fci_res['skeleton']
    adj_pc = None
    try:
        pc_res = run_pc(np, work_tr, var_cols, args.alpha, os.path.join(args.outdir, 'pc_graph.png'), os.path.join(args.outdir, 'pc_adjacency.csv'), debug_path=os.path.join(args.outdir, 'pc_debug.txt'))
        if args.adjacency_mode == 'directed':
            adj_pc = pc_res['adj']
        elif args.adjacency_mode == 'semi':
            adj_pc = pc_res['semi']
        else:
            adj_pc = pc_res['skeleton']
        if fs_mode == 'directed':
            adj_pc_fs = pc_res['adj']
        elif fs_mode == 'semi':
            adj_pc_fs = pc_res['semi']
        else:
            adj_pc_fs = pc_res['skeleton']
    except Exception as e:
        print(f'PC运行失败：{e}', file=sys.stderr)
    target_idx = var_cols.index(args.target_col)
    Xtr = work_tr.to_numpy(dtype=float)
    Xva = work_va.to_numpy(dtype=float)
    Xtr = np.where(np.isnan(Xtr), 0.0, Xtr)
    Xva = np.where(np.isnan(Xva), 0.0, Xva)
    ytr = Xtr[:, target_idx].copy()
    yva = Xva[:, target_idx].copy()
    Xtr[:, target_idx] = 0.0
    Xva[:, target_idx] = 0.0
    Xtr, meanX, stdX = zscore(pd, np, Xtr)
    Xva = (Xva - meanX) / stdX
    ytr, meany, stdy = zscore(pd, np, ytr.reshape(-1, 1))
    ytr = ytr.reshape(-1)
    yva = (yva.reshape(-1, 1) - meany) / (stdy + 1e-08)
    yva = yva.reshape(-1)
    results = []
    edge_counts = {}
    graph_feat_counts = {}
    if getattr(args, 'ivae_enabled', False):
        try:
            print('[iVAE] 启用 iVAE 实验组…')
            soft_u = False
            if args.ivae_u_type in ('station', 'plot'):
                u_col_candidates = []
                if args.ivae_u_type == 'plot':
                    u_col_candidates = ['样地代码', 'plot', 'Plot', 'plot_id']
                else:
                    u_col_candidates = ['生态站代码', 'station', 'Station', 'station_id']
                if args.group_col not in u_col_candidates:
                    u_col_candidates.append(args.group_col)
                u_col = None
                for c in u_col_candidates:
                    if c in df.columns:
                        u_col = c
                        break
                if u_col is None:
                    raise RuntimeError(f'未找到合适的U列，请检查数据列名（尝试: {u_col_candidates}）')
                u_ids_all, u_enc, n_levels = _encode_u_with_unk(pd, df[u_col], train_idx_eff)
                u_tr = u_ids_all.loc[train_idx_eff].to_numpy()
                u_va = u_ids_all.loc[val_idx_eff].to_numpy()
            else:
                K = int(args.vaed_clusters)
                z_dim = int(args.vaed_latent_dim)
                hidden = int(args.vaed_hidden)
                epochs = int(args.vaed_epochs)
                lr = float(args.vaed_lr)
                lambda3 = float(args.vaed_lambda3)
                gmm_update_every = int(args.vaed_gmm_update)
                vaed_p = _vaed_train_and_cluster(torch, np, Xtr, K=K, z_dim=z_dim, hidden=hidden, epochs=epochs, lr=lr, lambda3=lambda3, gmm_update_every=gmm_update_every, seed=42, station_labels=None, outdir=args.outdir)
                vaed_model = vaed_p.get('model', None)
                gmm = vaed_p['gmm']
                resp_tr = vaed_p['resp']
                if vaed_model is None:
                    print('[iVAE] 警告：VAED未返回模型，验证集伪U将退回均匀分布', file=sys.stderr)
                    resp_va = np.ones((Xva.shape[0], K), dtype=np.float32) / float(K)
                else:
                    vaed_model.eval()
                    with torch.no_grad():
                        Xva_t_local = torch.tensor(Xva, dtype=torch.float32, device=torch.device('cpu'))
                        _, mu_va_t, logvar_va_t, _ = vaed_model(Xva_t_local)
                    mu_va = mu_va_t.cpu().numpy()
                    if args.ivae_u_type == 'avaed':
                        resp_tr = _compute_attention_responsibilities(np, vaed_p['mu'], gmm, gamma=0.5, beta=1.0)
                        resp_va = _compute_attention_responsibilities(np, mu_va, gmm, gamma=0.5, beta=1.0)
                    else:
                        resp_va = _gmm_responsibilities_from_mu(np, mu_va, gmm)
                u_tr = resp_tr
                u_va = resp_va
                n_levels = K
                soft_u = True
            iVAEClass = _load_ivae_class()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = iVAEClass(data_dim=Xtr.shape[1], latent_dim=int(args.ivae_latent_dim), n_stations=int(n_levels), embedding_dim=int(args.ivae_embedding), hidden_dims=list(map(int, args.ivae_hidden)), activation='lrelu', device=str(device)).to(device)
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
            Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
            if soft_u:
                Utr_t = torch.tensor(u_tr, dtype=torch.float32, device=device)
                Uva_t = torch.tensor(u_va, dtype=torch.float32, device=device)
            else:
                Utr_t = torch.tensor(u_tr, dtype=torch.long, device=device)
                Uva_t = torch.tensor(u_va, dtype=torch.long, device=device)
            opt = torch.optim.Adam(model.parameters(), lr=float(args.ivae_lr))
            lambda_dag = float(args.ivae_lambda_dag)
            lambda_sparse = float(args.ivae_lambda_sparse)
            c = float(args.ivae_c_init)
            rho = float(args.ivae_rho)
            h_tol = float(args.ivae_h_tol)
            al_iters = int(args.ivae_al_iters)
            bs = int(args.ivae_batch_size)
            kl_warm_iters = int(args.ivae_kl_warmup_iters)
            kl_start = float(args.ivae_kl_start)
            kl_end = float(args.ivae_kl_end)
            Ntr = Xtr_t.size(0)
            num_batches = max(1, (Ntr + bs - 1) // bs)
            for it in range(al_iters):
                model.train()
                if kl_warm_iters > 0:
                    kl_coef = kl_start + (kl_end - kl_start) * min(1.0, it / kl_warm_iters)
                else:
                    kl_coef = kl_end
                perm = torch.randperm(Ntr, device=device)
                total_loss = 0.0
                for b in range(num_batches):
                    idx = perm[b * bs:(b + 1) * bs]
                    xb = Xtr_t[idx]
                    ub = Utr_t[idx]
                    loss, recon, kl, _ = model.elbo_loss(xb, ub)
                    hA = model.dag_constraint()
                    l1 = model.l1_regularization()
                    aug = recon + kl_coef * kl + lambda_dag * hA + 0.5 * c * (hA * hA) + lambda_sparse * l1
                    opt.zero_grad()
                    aug.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total_loss += float(aug.detach().cpu())
                with torch.no_grad():
                    hA_val = float(model.dag_constraint().detach().cpu())
                if (it + 1) % 10 == 0:
                    print(f'[iVAE] it={it + 1}/{al_iters} loss={total_loss / num_batches:.4f} hA={hA_val:.2e} kl_coef={kl_coef:.2f}')
                if abs(hA_val) <= h_tol:
                    lambda_sparse = lambda_sparse
                else:
                    c *= rho
            Ztr = model.get_latent_representations(Xtr_t, Utr_t)
            Zva = model.get_latent_representations(Xva_t, Uva_t)
            A_cont = model.get_adjacency_matrix()
            try:
                import pandas as _pd
                base_thr = float(args.ivae_graph_threshold)
                tgt_edges = int(getattr(args, 'ivae_target_edges', 0) or 0)
                A_bin, thr_eff, _ = _binarize_adj_with_budget(np, A_cont, base_thr, tgt_edges)
                z_labels, z_details = _compute_z_alias_labels(np, Xtr, Ztr, var_cols, top_k=3)
                _save_z_alias_csv(_pd, z_details, os.path.join(args.outdir, 'ivae_z_alias.csv'))
                base = os.path.join(args.outdir, os.path.splitext(args.ivae_adj_out)[0])
                _pd.DataFrame(np.asarray(A_cont), dtype=float).to_csv(base + '_continuous.csv', index=False, header=False, encoding='utf-8-sig')
                _pd.DataFrame(A_bin, dtype=int).to_csv(os.path.join(args.outdir, args.ivae_adj_out), index=False, header=False, encoding='utf-8-sig')
                _plot_latent_causal_graph(np, nx, plt, A_bin, os.path.join(args.outdir, args.ivae_graph_png), labels_override=z_labels)
                ecount_ivae = _edge_count_from_adj(np, A_bin)
                edge_counts['IVAE_Z'] = ecount_ivae
                edge_counts['IVAE_X'] = ecount_ivae
                edge_counts['IVAE_XZ'] = ecount_ivae
                print(f'[iVAE] 阈值= {thr_eff:.4g} 边数= {ecount_ivae}')
            except Exception as e:
                print(f'iVAE 邻接保存/绘图失败：{e}', file=sys.stderr)

            def run_triad(tag_prefix: str, variant_label: str, Xtr_src, Xva_src):
                nonlocal results
                try:
                    _, _, mtr, mva = train_and_eval_lgb(np, Xtr_src, ytr, Xva_src, yva, args.outdir, tag_prefix)
                    results.append((variant_label, 'LightGBM', edge_counts.get(variant_label, None), *mtr, *mva))
                except Exception as e:
                    print(f'LightGBM({tag_prefix})训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_xgb(np, Xtr_src, ytr, Xva_src, yva, args.outdir, tag_prefix)
                    results.append((variant_label, 'XGBoost', edge_counts.get(variant_label, None), *mtr, *mva))
                except Exception as e:
                    print(f'XGBoost({tag_prefix})训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_src, ytr, Xva_src, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, tag_prefix)
                    results.append((variant_label, 'MLP', edge_counts.get(variant_label, None), *mtr, *mva))
                except Exception as e:
                    print(f'MLP({tag_prefix})训练失败：{e}', file=sys.stderr)
            run_triad('soc_IVAE_Z', 'IVAE_Z', Ztr, Zva)
            run_triad('soc_IVAE_X', 'IVAE_X', Xtr, Xva)
            XZtr = np.concatenate([Xtr, Ztr], axis=1)
            XZva = np.concatenate([Xva, Zva], axis=1)
            run_triad('soc_IVAE_XZ', 'IVAE_XZ', XZtr, XZva)
            try:
                if getattr(args, 'gf_enabled', False):
                    A_core = A_bin
                    if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                        A_core = (A_core + A_core.T > 0).astype(int)
                    GFZ_tr = _graph_poly_features(np, Ztr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                    GFZ_va = _graph_poly_features(np, Zva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                    try:
                        graph_feat_counts['IVAE_Z_GF'] = int(GFZ_tr.shape[1])
                    except Exception:
                        pass
                    if args.gf_attach == 'concat':
                        Ztr_gf = np.concatenate([Ztr, GFZ_tr], axis=1)
                        Zva_gf = np.concatenate([Zva, GFZ_va], axis=1)
                    else:
                        Ztr_gf, Zva_gf = (GFZ_tr, GFZ_va)
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, Ztr_gf, ytr, Zva_gf, yva, args.outdir, 'soc_IVAE_Z_GF')
                        results.append(('IVAE_Z_GF', 'LightGBM', edge_counts.get('IVAE_Z', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(soc_IVAE_Z_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, Ztr_gf, ytr, Zva_gf, yva, args.outdir, 'soc_IVAE_Z_GF')
                        results.append(('IVAE_Z_GF', 'XGBoost', edge_counts.get('IVAE_Z', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(soc_IVAE_Z_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, Ztr_gf, ytr, Zva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_IVAE_Z_GF')
                        results.append(('IVAE_Z_GF', 'MLP', edge_counts.get('IVAE_Z', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(soc_IVAE_Z_GF)训练失败：{e}', file=sys.stderr)
                    if args.gf_attach == 'concat':
                        Xtr_gf = np.concatenate([Xtr, GFZ_tr], axis=1)
                        Xva_gf = np.concatenate([Xva, GFZ_va], axis=1)
                    else:
                        Xtr_gf, Xva_gf = (GFZ_tr, GFZ_va)
                    try:
                        graph_feat_counts['IVAE_X_GF'] = int(GFZ_tr.shape[1])
                    except Exception:
                        pass
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_IVAE_X_GF')
                        results.append(('IVAE_X_GF', 'LightGBM', edge_counts.get('IVAE_X', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(soc_IVAE_X_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_IVAE_X_GF')
                        results.append(('IVAE_X_GF', 'XGBoost', edge_counts.get('IVAE_X', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(soc_IVAE_X_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_IVAE_X_GF')
                        results.append(('IVAE_X_GF', 'MLP', edge_counts.get('IVAE_X', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(soc_IVAE_X_GF)训练失败：{e}', file=sys.stderr)
                    if args.gf_attach == 'concat':
                        XZtr_gf = np.concatenate([XZtr, GFZ_tr], axis=1)
                        XZva_gf = np.concatenate([XZva, GFZ_va], axis=1)
                    else:
                        XZtr_gf, XZva_gf = (GFZ_tr, GFZ_va)
                    try:
                        graph_feat_counts['IVAE_XZ_GF'] = int(GFZ_tr.shape[1])
                    except Exception:
                        pass
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, XZtr_gf, ytr, XZva_gf, yva, args.outdir, 'soc_IVAE_XZ_GF')
                        results.append(('IVAE_XZ_GF', 'LightGBM', edge_counts.get('IVAE_XZ', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(soc_IVAE_XZ_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, XZtr_gf, ytr, XZva_gf, yva, args.outdir, 'soc_IVAE_XZ_GF')
                        results.append(('IVAE_XZ_GF', 'XGBoost', edge_counts.get('IVAE_XZ', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(soc_IVAE_XZ_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, XZtr_gf, ytr, XZva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_IVAE_XZ_GF')
                        results.append(('IVAE_XZ_GF', 'MLP', edge_counts.get('IVAE_XZ', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(soc_IVAE_XZ_GF)训练失败：{e}', file=sys.stderr)
            except Exception as e:
                print(f'iVAE GF 流程失败：{e}', file=sys.stderr)
            print('[iVAE] 完成 iVAE 实验组')
        except Exception as e:
            print(f'iVAE 实验组执行失败：{e}', file=sys.stderr)
    try:
        sel_idx = select_features_by_adj(np, adj_fci_fs, target_idx, mode='nbr', fallback_to_all=False)
        try:
            edge_counts['FCI'] = _edge_count_from_adj(np, adj_fci)
        except Exception:
            pass
        Xtr_sel = Xtr[:, sel_idx]
        Xva_sel = Xva[:, sel_idx]
        if Xtr_sel.shape[1] == 0:
            print('[FCI] 特征选择为空，跳过表格模型(LGB/XGB/MLP)以避免零特征错误；建议使用 --fs-adjacency-mode skeleton 或启用 --gf-enabled', file=sys.stderr)
        else:
            try:
                _, _, mtr, mva = train_and_eval_lgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_FCI')
                results.append(('FCI', 'LightGBM', edge_counts.get('FCI', None), *mtr, *mva))
            except Exception as e:
                print(f'LightGBM(FCI)训练失败：{e}', file=sys.stderr)
            try:
                _, _, mtr, mva = train_and_eval_xgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_FCI')
                results.append(('FCI', 'XGBoost', edge_counts.get('FCI', None), *mtr, *mva))
            except Exception as e:
                print(f'XGBoost(FCI)训练失败：{e}', file=sys.stderr)
            try:
                _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_sel, ytr, Xva_sel, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_FCI')
                results.append(('FCI', 'MLP', edge_counts.get('FCI', None), *mtr, *mva))
            except Exception as e:
                print(f'MLP(FCI)训练失败：{e}', file=sys.stderr)
        try:
            if getattr(args, 'gf_enabled', False):
                A_core = adj_fci_fs
                GF_tr = _graph_poly_features(np, Xtr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                GF_va = _graph_poly_features(np, Xva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                try:
                    graph_feat_counts['FCI_GF'] = int(GF_tr.shape[1])
                except Exception:
                    pass
                if args.gf_attach == 'concat':
                    Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                    Xva_gf = np.concatenate([Xva, GF_va], axis=1)
                else:
                    Xtr_gf, Xva_gf = (GF_tr, GF_va)
                try:
                    _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_FCI_GF')
                    results.append(('FCI_GF', 'LightGBM', edge_counts.get('FCI', None), *mtr, *mva))
                except Exception as e:
                    print(f'LightGBM(FCI_GF)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_FCI_GF')
                    results.append(('FCI_GF', 'XGBoost', edge_counts.get('FCI', None), *mtr, *mva))
                except Exception as e:
                    print(f'XGBoost(FCI_GF)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_FCI_GF')
                    results.append(('FCI_GF', 'MLP', edge_counts.get('FCI', None), *mtr, *mva))
                except Exception as e:
                    print(f'MLP(FCI_GF)训练失败：{e}', file=sys.stderr)
        except Exception as e:
            print(f'FCI 图滤波特征失败：{e}', file=sys.stderr)
    except Exception as e:
        print(f'FCI 下游(LGB/XGB/MLP)失败：{e}', file=sys.stderr)
    adj_fci = make_gnn_adj(adj_fci, symmetrize=args.symmetrize_adj, add_self_loop=True)
    if args.augment_target_k and args.augment_target_k > 0:
        Xtr_full = work_tr.to_numpy(dtype=float)
        Xtr_full = np.where(np.isnan(Xtr_full), 0.0, Xtr_full)
        ytr_full = Xtr_full[:, target_idx].copy()
        adj_fci = augment_adj_with_target(np, adj_fci, Xtr_full, ytr_full, target_idx, args.augment_target_k)
    if adj_pc is not None:
        adj_pc_bin = make_gnn_adj(adj_pc, symmetrize=args.symmetrize_adj, add_self_loop=True)
        try:
            edge_counts['PC'] = _edge_count_from_adj(np, adj_pc)
        except Exception:
            pass
        if args.augment_target_k and args.augment_target_k > 0:
            Xtr_full = work_tr.to_numpy(dtype=float)
            Xtr_full = np.where(np.isnan(Xtr_full), 0.0, Xtr_full)
            ytr_full = Xtr_full[:, target_idx].copy()
            adj_pc_bin = augment_adj_with_target(np, adj_pc_bin, Xtr_full, ytr_full, target_idx, args.augment_target_k)
    else:
        adj_pc_bin = None
    try:
        _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GraphSAGE', adj_fci, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_FCI')
        results.append(('FCI', 'GraphSAGE', edge_counts.get('FCI', None), *mtr, *mva))
    except Exception as e:
        print(f'GraphSAGE(FCI)训练失败：{e}', file=sys.stderr)
    try:
        _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GCN', adj_fci, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_FCI')
        results.append(('FCI', 'GCN', edge_counts.get('FCI', None), *mtr, *mva))
    except Exception as e:
        print(f'GCN(FCI)训练失败：{e}', file=sys.stderr)
    try:
        _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'ECMPNN', adj_fci, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_FCI')
        results.append(('FCI', 'ECMPNN', edge_counts.get('FCI', None), *mtr, *mva))
    except Exception as e:
        print(f'ECMPNN(FCI)训练失败：{e}', file=sys.stderr)
    if adj_pc_bin is not None:
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GraphSAGE', adj_pc_bin, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_PC')
            results.append(('PC', 'GraphSAGE', edge_counts.get('PC', None), *mtr, *mva))
        except Exception as e:
            print(f'GraphSAGE(PC)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GCN', adj_pc_bin, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_PC')
            results.append(('PC', 'GCN', edge_counts.get('PC', None), *mtr, *mva))
        except Exception as e:
            print(f'GCN(PC)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'ECMPNN', adj_pc_bin, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_PC')
            results.append(('PC', 'ECMPNN', edge_counts.get('PC', None), *mtr, *mva))
        except Exception as e:
            print(f'ECMPNN(PC)训练失败：{e}', file=sys.stderr)
        try:
            if getattr(args, 'gf_enabled', False) and 'adj_pc_fs' in locals():
                A_core = adj_pc_fs
                GF_tr = _graph_poly_features(np, Xtr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                GF_va = _graph_poly_features(np, Xva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                try:
                    graph_feat_counts['PC_GF'] = int(GF_tr.shape[1])
                except Exception:
                    pass
                if args.gf_attach == 'concat':
                    Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                    Xva_gf = np.concatenate([Xva, GF_va], axis=1)
                else:
                    Xtr_gf, Xva_gf = (GF_tr, GF_va)
                try:
                    _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_PC_GF')
                    results.append(('PC_GF', 'LightGBM', edge_counts.get('PC', None), *mtr, *mva))
                except Exception as e:
                    print(f'LightGBM(PC_GF)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_PC_GF')
                    results.append(('PC_GF', 'XGBoost', edge_counts.get('PC', None), *mtr, *mva))
                except Exception as e:
                    print(f'XGBoost(PC_GF)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_PC_GF')
                    results.append(('PC_GF', 'MLP', edge_counts.get('PC', None), *mtr, *mva))
                except Exception as e:
                    print(f'MLP(PC_GF)训练失败：{e}', file=sys.stderr)
        except Exception as e:
            print(f'PC 图滤波特征失败：{e}', file=sys.stderr)
    try:
        adj_dag = run_dag_gnn(torch, np, work_tr, var_cols, hidden=args.dag_hidden, epochs=args.dag_epochs, lr=args.dag_lr, lambda_acyc=args.dag_lambda_acyc, lambda_sparse=args.dag_lambda_sparse, threshold=args.dag_threshold, out_png=os.path.join(args.outdir, args.dag_graph_png), out_adj=os.path.join(args.outdir, args.dag_adj_out), edges_dbg_out=os.path.join(args.outdir, args.dag_edges_out), cat_cols=list(encoders.keys()))
        try:
            edge_counts['DAGGNN'] = _edge_count_from_adj(np, adj_dag)
        except Exception:
            pass
        adj_dag_core = adj_dag.copy()
        adj_dag = make_gnn_adj(adj_dag_core, symmetrize=args.symmetrize_adj, add_self_loop=True)
        if args.augment_target_k and args.augment_target_k > 0:
            Xtr_full = work_tr.to_numpy(dtype=float)
            Xtr_full = np.where(np.isnan(Xtr_full), 0.0, Xtr_full)
            ytr_full = Xtr_full[:, target_idx].copy()
            adj_dag = augment_adj_with_target(np, adj_dag, Xtr_full, ytr_full, target_idx, args.augment_target_k)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GraphSAGE', adj_dag, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_DAG')
            results.append(('DAGGNN', 'GraphSAGE', edge_counts.get('DAGGNN', None), *mtr, *mva))
        except Exception as e:
            print(f'GraphSAGE(DAG-GNN)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GCN', adj_dag, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_DAG')
            results.append(('DAGGNN', 'GCN', edge_counts.get('DAGGNN', None), *mtr, *mva))
        except Exception as e:
            print(f'GCN(DAG-GNN)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'ECMPNN', adj_dag, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_DAG')
            results.append(('DAGGNN', 'ECMPNN', edge_counts.get('DAGGNN', None), *mtr, *mva))
        except Exception as e:
            print(f'ECMPNN(DAG-GNN)训练失败：{e}', file=sys.stderr)
        if not getattr(args, 'gf_enabled', False):
            try:
                if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                    adj_dag_fs = (adj_dag_core + adj_dag_core.T > 0).astype(int)
                else:
                    adj_dag_fs = adj_dag_core
                sel_idx = select_features_by_adj(np, adj_dag_fs, target_idx, mode='nbr', fallback_to_all=False)
                Xtr_sel = Xtr[:, sel_idx]
                Xva_sel = Xva[:, sel_idx]
                if Xtr_sel.shape[1] == 0:
                    print('[DAGGNN] 特征选择为空，跳过表格模型(LGB/XGB/MLP)以避免零特征错误；建议使用 --fs-adjacency-mode skeleton 或启用 --gf-enabled', file=sys.stderr)
                else:
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_DAG')
                        results.append(('DAGGNN', 'LightGBM', edge_counts.get('DAGGNN', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(DAG)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_DAG')
                        results.append(('DAGGNN', 'XGBoost', edge_counts.get('DAGGNN', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(DAG)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_sel, ytr, Xva_sel, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_DAG')
                        results.append(('DAGGNN', 'MLP', edge_counts.get('DAGGNN', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(DAG)训练失败：{e}', file=sys.stderr)
            except Exception as e:
                print(f'DAG 下游(LGB/XGB/MLP)失败：{e}', file=sys.stderr)
            try:
                if getattr(args, 'gf_enabled', False):
                    A_core = adj_dag_core
                    if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                        A_core = (A_core + A_core.T > 0).astype(int)
                    GF_tr = _graph_poly_features(np, Xtr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                    GF_va = _graph_poly_features(np, Xva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                    try:
                        graph_feat_counts['DAGGNN_GF'] = int(GF_tr.shape[1])
                    except Exception:
                        pass
                    if args.gf_attach == 'concat':
                        Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                        Xva_gf = np.concatenate([Xva, GF_va], axis=1)
                    else:
                        Xtr_gf, Xva_gf = (GF_tr, GF_va)
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_DAG_GF')
                        results.append(('DAGGNN_GF', 'LightGBM', edge_counts.get('DAGGNN', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_DAG_GF')
                        results.append(('DAGGNN_GF', 'XGBoost', edge_counts.get('DAGGNN', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_DAG_GF')
                        results.append(('DAGGNN_GF', 'MLP', edge_counts.get('DAGGNN', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(DAG_GF)训练失败：{e}', file=sys.stderr)
            except Exception as e:
                print(f'DAG 图滤波特征失败：{e}', file=sys.stderr)
    except Exception as e:
        print(f'DAG-GNN 结构学习失败：{e}', file=sys.stderr)
    try:
        N = adj_fci.shape[0]
        if getattr(args, 'rand_edges', 0) is not None and args.rand_edges > 0:
            m_edges = int(args.rand_edges)
        else:
            skel = fci_res.get('skeleton', (adj_fci + adj_fci.T > 0).astype(int))
            m_edges = int(np.triu(skel, 1).sum())
        rand_res = random_endpoint_graph(np, N, m_edges, seed=int(args.rand_seed))
        base = os.path.join(args.outdir, 'rand_adjacency')
        import pandas as pd
        pd.DataFrame(rand_res['tail'], index=var_cols, columns=var_cols).to_csv(base + '_tail.csv', encoding='utf-8-sig')
        pd.DataFrame(rand_res['arrow'], index=var_cols, columns=var_cols).to_csv(base + '_arrow.csv', encoding='utf-8-sig')
        pd.DataFrame(rand_res['circle'], index=var_cols, columns=var_cols).to_csv(base + '_circle.csv', encoding='utf-8-sig')
        pd.DataFrame(rand_res['adj'], index=var_cols, columns=var_cols).to_csv(base + '.csv', encoding='utf-8-sig')
        pd.DataFrame(rand_res['skeleton'], index=var_cols, columns=var_cols).to_csv(base + '_skeleton.csv', encoding='utf-8-sig')
        pd.DataFrame(rand_res['semi'], index=var_cols, columns=var_cols).to_csv(base + '_semi.csv', encoding='utf-8-sig')
        if args.adjacency_mode == 'directed':
            rand_adj_core = rand_res['adj']
        elif args.adjacency_mode == 'semi':
            rand_adj_core = rand_res['semi']
        else:
            rand_adj_core = rand_res['skeleton']
        try:
            edge_counts['RAND'] = _edge_count_from_adj(np, rand_adj_core)
        except Exception:
            pass
        rand_adj = make_gnn_adj(rand_adj_core, symmetrize=args.symmetrize_adj, add_self_loop=True)
        if args.augment_target_k and args.augment_target_k > 0:
            Xtr_full = work_tr.to_numpy(dtype=float)
            Xtr_full = np.where(np.isnan(Xtr_full), 0.0, Xtr_full)
            ytr_full = Xtr_full[:, target_idx].copy()
            rand_adj = augment_adj_with_target(np, rand_adj, Xtr_full, ytr_full, target_idx, args.augment_target_k)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GraphSAGE', rand_adj, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_RAND')
            results.append(('RAND', 'GraphSAGE', edge_counts.get('RAND', None), *mtr, *mva))
        except Exception as e:
            print(f'GraphSAGE(RAND)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GCN', rand_adj, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_RAND')
            results.append(('RAND', 'GCN', edge_counts.get('RAND', None), *mtr, *mva))
        except Exception as e:
            print(f'GCN(RAND)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'ECMPNN', rand_adj, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_RAND')
            results.append(('RAND', 'ECMPNN', edge_counts.get('RAND', None), *mtr, *mva))
        except Exception as e:
            print(f'ECMPNN(RAND)训练失败：{e}', file=sys.stderr)
        try:
            if getattr(args, 'gf_enabled', False):
                A_core = rand_adj_core
                if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                    A_core = (A_core + A_core.T > 0).astype(int)
                GF_tr = _graph_poly_features(np, Xtr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                GF_va = _graph_poly_features(np, Xva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                try:
                    graph_feat_counts['RAND_GF'] = int(GF_tr.shape[1])
                except Exception:
                    pass
                if args.gf_attach == 'concat':
                    Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                    Xva_gf = np.concatenate([Xva, GF_va], axis=1)
                else:
                    Xtr_gf, Xva_gf = (GF_tr, GF_va)
                try:
                    _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_RAND_GF')
                    results.append(('RAND_GF', 'LightGBM', edge_counts.get('RAND', None), *mtr, *mva))
                except Exception as e:
                    print(f'LightGBM(RAND_GF)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_RAND_GF')
                    results.append(('RAND_GF', 'XGBoost', edge_counts.get('RAND', None), *mtr, *mva))
                except Exception as e:
                    print(f'XGBoost(RAND_GF)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_RAND_GF')
                    results.append(('RAND_GF', 'MLP', edge_counts.get('RAND', None), *mtr, *mva))
                except Exception as e:
                    print(f'MLP(RAND_GF)训练失败：{e}', file=sys.stderr)
        except Exception as e:
            print(f'RAND 图滤波特征失败：{e}', file=sys.stderr)
    except Exception as e:
        print(f'随机图对照失败：{e}', file=sys.stderr)
    try:
        N = Xtr.shape[1]
        full_core = np.ones((N, N), dtype=int)
        np.fill_diagonal(full_core, 0)
        try:
            edge_counts['FULL'] = _edge_count_from_adj(np, full_core)
        except Exception:
            pass
        GF_tr = _graph_poly_features(np, Xtr, full_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
        GF_va = _graph_poly_features(np, Xva, full_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
        try:
            graph_feat_counts['FULL_GF'] = int(GF_tr.shape[1])
        except Exception:
            pass
        Xtr_full = np.concatenate([Xtr, GF_tr], axis=1) if args.gf_attach == 'concat' or True else GF_tr
        Xva_full = np.concatenate([Xva, GF_va], axis=1) if args.gf_attach == 'concat' or True else GF_va
        try:
            _, _, mtr, mva = train_and_eval_lgb(np, Xtr_full, ytr, Xva_full, yva, args.outdir, 'soc_FULL_GF')
            results.append(('FULL_GF', 'LightGBM', edge_counts.get('FULL', None), *mtr, *mva))
        except Exception as e:
            print(f'LightGBM(FULL_GF)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_xgb(np, Xtr_full, ytr, Xva_full, yva, args.outdir, 'soc_FULL_GF')
            results.append(('FULL_GF', 'XGBoost', edge_counts.get('FULL', None), *mtr, *mva))
        except Exception as e:
            print(f'XGBoost(FULL_GF)训练失败：{e}', file=sys.stderr)
        try:
            _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_full, ytr, Xva_full, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_FULL_GF')
            results.append(('FULL_GF', 'MLP', edge_counts.get('FULL', None), *mtr, *mva))
        except Exception as e:
            print(f'MLP(FULL_GF)训练失败：{e}', file=sys.stderr)
    except Exception as e:
        print(f'FULL_GF 下游(LGB/XGB/MLP)失败：{e}', file=sys.stderr)
    if args.vaed_enabled:
        try:
            X_tr_np = work_tr.to_numpy(dtype=float)
            X_tr_np = np.where(np.isnan(X_tr_np), 0.0, X_tr_np)
            X_tr_s, mXv, sXv = zscore(pd, np, X_tr_np)
            station_labels = tr_df_full[args.group_col].loc[train_idx_eff].to_numpy()
            vaed_res = _vaed_train_and_cluster(torch, np, X_tr_s, K=int(args.vaed_clusters), z_dim=int(args.vaed_latent_dim), hidden=int(args.vaed_hidden), epochs=int(args.vaed_epochs), lr=float(args.vaed_lr), lambda3=float(args.vaed_lambda3), gmm_update_every=int(args.vaed_gmm_update), seed=42, station_labels=station_labels, outdir=args.outdir)
            assign = vaed_res['resp'].argmax(axis=1)
            counts = [(k, int((assign == k).sum())) for k in range(int(args.vaed_clusters))]
            print('[VAED] final cluster sizes: ' + ', '.join([f'k{k}={v}' for k, v in counts]))
            try:
                _plot_cluster_station_stack(pd, np, vaed_res['resp'], station_labels, args.outdir, fname='vaed_cluster_station_stack.png')
            except Exception:
                pass
            try:
                _plot_cluster_feature_stats(pd, np, vaed_res['resp'], work_tr, var_cols, args.target_col, args.outdir, top_n=8, fname_radar='vaed_cluster_feature_radar.png', fname_bars='vaed_cluster_feature_bars.png')
            except Exception:
                pass
            dag_runner_ctx = {'cat_cols': list(encoders.keys())}
            A_vaed, A_vaed_soft_cont, A_vaed_soft_prop = _aggregate_cluster_dags(np, var_cols, work_tr, vaed_res['resp'], args, dag_runner_ctx, agg_threshold=float(args.vaed_agg_threshold), prefix='vaed')
            try:
                import pandas as _pd
                thr_vaed_eff = float(args.vaed_agg_threshold)
                try:
                    tgt_edges = int(getattr(args, 'agg_target_edges', 0) or 0)
                except Exception:
                    tgt_edges = 0
                if tgt_edges > 0:
                    A_vaed, thr_vaed_eff, _ = _binarize_adj_with_budget(np, A_vaed_soft_prop, base_threshold=thr_vaed_eff, target_edges=tgt_edges)
                _plot_variable_causal_graph(np, nx, plt, A_vaed, var_cols, os.path.join(args.outdir, 'vaed_dag_graph.png'))
                _pd.DataFrame(A_vaed, index=var_cols, columns=var_cols).to_csv(os.path.join(args.outdir, 'vaed_dag_adj.csv'), encoding='utf-8-sig')
                _pd.DataFrame(A_vaed_soft_cont, index=var_cols, columns=var_cols).to_csv(os.path.join(args.outdir, 'vaed_soft_adj_continuous.csv'), encoding='utf-8-sig')
                _pd.DataFrame(A_vaed_soft_prop, index=var_cols, columns=var_cols).to_csv(os.path.join(args.outdir, 'vaed_soft_adj_proportion.csv'), encoding='utf-8-sig')
                edge_counts['VAED_DAG'] = _edge_count_from_adj(np, A_vaed)
                print(f"[VAED] 阈值= {thr_vaed_eff:.4g} 边数= {edge_counts['VAED_DAG']}")
            except Exception as e:
                print(f'VAED_DAG 绘图/保存失败：{e}', file=sys.stderr)
            A_vaed_core = A_vaed.copy()
            A_vaed = make_gnn_adj(A_vaed_core, symmetrize=args.symmetrize_adj, add_self_loop=True)
            if args.augment_target_k and args.augment_target_k > 0:
                Xtr_full = work_tr.to_numpy(dtype=float)
                Xtr_full = np.where(np.isnan(Xtr_full), 0.0, Xtr_full)
                ytr_full = Xtr_full[:, target_idx].copy()
                A_vaed = augment_adj_with_target(np, A_vaed, Xtr_full, ytr_full, target_idx, args.augment_target_k)
            try:
                _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GraphSAGE', A_vaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_VAED_DAG')
                results.append(('VAED_DAG', 'GraphSAGE', edge_counts.get('VAED_DAG', None), *mtr, *mva))
            except Exception as e:
                print(f'GraphSAGE(VAED_DAG)训练失败：{e}', file=sys.stderr)
            try:
                _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GCN', A_vaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_VAED_DAG')
                results.append(('VAED_DAG', 'GCN', edge_counts.get('VAED_DAG', None), *mtr, *mva))
            except Exception as e:
                print(f'GCN(VAED_DAG)训练失败：{e}', file=sys.stderr)
            try:
                _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'ECMPNN', A_vaed, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_VAED_DAG')
                results.append(('VAED_DAG', 'ECMPNN', edge_counts.get('VAED_DAG', None), *mtr, *mva))
            except Exception as e:
                print(f'ECMPNN(VAED_DAG)训练失败：{e}', file=sys.stderr)
            try:
                attn = _compute_attention_responsibilities(np, vaed_res['mu'], vaed_res['gmm'], gamma=0.5, beta=1.0, temp_fine=1.0, temp_coarse=2.0)
                try:
                    _plot_cluster_station_stack(pd, np, attn, station_labels, args.outdir, fname='avaed_cluster_station_stack.png')
                except Exception:
                    pass
                try:
                    _plot_cluster_feature_stats(pd, np, attn, work_tr, var_cols, args.target_col, args.outdir, top_n=8, fname_radar='avaed_cluster_feature_radar.png', fname_bars='avaed_cluster_feature_bars.png')
                except Exception:
                    pass
                dag_runner_ctx = {'cat_cols': list(encoders.keys())}
                A_avaed, A_avaed_soft_cont, A_avaed_soft_prop = _aggregate_cluster_dags(np, var_cols, work_tr, attn, args, dag_runner_ctx, agg_threshold=float(getattr(args, 'avaed_agg_threshold', args.vaed_agg_threshold)), prefix='avaed')
                try:
                    import pandas as _pd
                    thr_avaed_eff = float(getattr(args, 'avaed_agg_threshold', args.vaed_agg_threshold))
                    try:
                        tgt_edges = int(getattr(args, 'agg_target_edges', 0) or 0)
                    except Exception:
                        tgt_edges = 0
                    if tgt_edges > 0:
                        A_avaed, thr_avaed_eff, _ = _binarize_adj_with_budget(np, A_avaed_soft_prop, base_threshold=thr_avaed_eff, target_edges=tgt_edges)
                    _plot_variable_causal_graph(np, nx, plt, A_avaed, var_cols, os.path.join(args.outdir, 'avaed_dag_graph.png'))
                    _pd.DataFrame(A_avaed, index=var_cols, columns=var_cols).to_csv(os.path.join(args.outdir, 'avaed_dag_adj.csv'), encoding='utf-8-sig')
                    _pd.DataFrame(A_avaed_soft_cont, index=var_cols, columns=var_cols).to_csv(os.path.join(args.outdir, 'avaed_soft_adj_continuous.csv'), encoding='utf-8-sig')
                    _pd.DataFrame(A_avaed_soft_prop, index=var_cols, columns=var_cols).to_csv(os.path.join(args.outdir, 'avaed_soft_adj_proportion.csv'), encoding='utf-8-sig')
                    edge_counts['AVAED_DAG'] = _edge_count_from_adj(np, A_avaed)
                    print(f"[AVAED] 阈值= {thr_avaed_eff:.4g} 边数= {edge_counts['AVAED_DAG']}")
                except Exception as e:
                    print(f'AVAED_DAG 绘图/保存失败：{e}', file=sys.stderr)
                A_avaed_core = A_avaed.copy()
                A_avaed = make_gnn_adj(A_avaed_core, symmetrize=args.symmetrize_adj, add_self_loop=True)
                if args.augment_target_k and args.augment_target_k > 0:
                    Xtr_full = work_tr.to_numpy(dtype=float)
                    Xtr_full = np.where(np.isnan(Xtr_full), 0.0, Xtr_full)
                    ytr_full = Xtr_full[:, target_idx].copy()
                    A_avaed = augment_adj_with_target(np, A_avaed, Xtr_full, ytr_full, target_idx, args.augment_target_k)
                try:
                    _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GraphSAGE', A_avaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_AVAED_DAG')
                    results.append(('AVAED_DAG', 'GraphSAGE', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                except Exception as e:
                    print(f'GraphSAGE(AVAED_DAG)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'GCN', A_avaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_AVAED_DAG')
                    results.append(('AVAED_DAG', 'GCN', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                except Exception as e:
                    print(f'GCN(AVAED_DAG)训练失败：{e}', file=sys.stderr)
                try:
                    _, _, mtr, mva = train_and_eval_model(torch, np, Xtr, ytr, Xva, yva, 'ECMPNN', A_avaed, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_AVAED_DAG')
                    results.append(('AVAED_DAG', 'ECMPNN', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                except Exception as e:
                    print(f'ECMPNN(AVAED_DAG)训练失败：{e}', file=sys.stderr)
                try:
                    if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                        A_avaed_fs = (A_avaed_core + A_avaed_core.T > 0).astype(int)
                    else:
                        A_avaed_fs = A_avaed_core
                    sel_idx = select_features_by_adj(np, A_avaed_fs, target_idx, mode='nbr', fallback_to_all=False)
                    Xtr_sel = Xtr[:, sel_idx]
                    Xva_sel = Xva[:, sel_idx]
                    if Xtr_sel.shape[1] == 0:
                        print('[AVAED_DAG] 特征选择为空，跳过表格模型(LGB/XGB/MLP)以避免零特征错误；建议使用 --fs-adjacency-mode skeleton 或启用 --gf-enabled', file=sys.stderr)
                    else:
                        try:
                            _, _, mtr, mva = train_and_eval_lgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_AVAED_DAG')
                            results.append(('AVAED_DAG', 'LightGBM', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'LightGBM(AVAED_DAG)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_xgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_AVAED_DAG')
                            results.append(('AVAED_DAG', 'XGBoost', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'XGBoost(AVAED_DAG)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_sel, ytr, Xva_sel, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_AVAED_DAG')
                            results.append(('AVAED_DAG', 'MLP', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'MLP(AVAED_DAG)训练失败：{e}', file=sys.stderr)
                except Exception as e:
                    print(f'AVAED_DAG 下游(LGB/XGB/MLP)失败：{e}', file=sys.stderr)
                try:
                    if getattr(args, 'gf_enabled', False):
                        A_core = A_avaed_core
                        if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                            A_core = (A_core + A_core.T > 0).astype(int)
                        GF_tr = _graph_poly_features(np, Xtr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                        GF_va = _graph_poly_features(np, Xva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                        try:
                            graph_feat_counts['AVAED_DAG_GF'] = int(GF_tr.shape[1])
                        except Exception:
                            pass
                        if args.gf_attach == 'concat':
                            Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                            Xva_gf = np.concatenate([Xva, GF_va], axis=1)
                        else:
                            Xtr_gf, Xva_gf = (GF_tr, GF_va)
                        try:
                            _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_AVAED_DAG_GF')
                            results.append(('AVAED_DAG_GF', 'LightGBM', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'LightGBM(AVAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_AVAED_DAG_GF')
                            results.append(('AVAED_DAG_GF', 'XGBoost', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'XGBoost(AVAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_AVAED_DAG_GF')
                            results.append(('AVAED_DAG_GF', 'MLP', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'MLP(AVAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'GraphSAGE', A_avaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_AVAED_DAG_GF')
                            results.append(('AVAED_DAG_GF', 'GraphSAGE', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'GraphSAGE(AVAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'GCN', A_avaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_AVAED_DAG_GF')
                            results.append(('AVAED_DAG_GF', 'GCN', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'GCN(AVAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                        try:
                            _, _, mtr, mva = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'ECMPNN', A_avaed, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_AVAED_DAG_GF')
                            results.append(('AVAED_DAG_GF', 'ECMPNN', edge_counts.get('AVAED_DAG', None), *mtr, *mva))
                        except Exception as e:
                            print(f'ECMPNN(AVAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                except Exception as e:
                    print(f'AVAED 图滤波特征失败：{e}', file=sys.stderr)
            except Exception as e:
                print(f'Attention-VAED 流程失败：{e}', file=sys.stderr)
            try:
                if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                    A_vaed_fs = (A_vaed_core + A_vaed_core.T > 0).astype(int)
                else:
                    A_vaed_fs = A_vaed_core
                sel_idx = select_features_by_adj(np, A_vaed_fs, target_idx, mode='nbr', fallback_to_all=False)
                Xtr_sel = Xtr[:, sel_idx]
                Xva_sel = Xva[:, sel_idx]
                if Xtr_sel.shape[1] == 0:
                    print('[VAED_DAG] 特征选择为空，跳过表格模型(LGB/XGB/MLP)以避免零特征错误；建议使用 --fs-adjacency-mode skeleton 或启用 --gf-enabled', file=sys.stderr)
                else:
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_VAED_DAG')
                        results.append(('VAED_DAG', 'LightGBM', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(VAED_DAG)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, Xtr_sel, ytr, Xva_sel, yva, args.outdir, 'soc_VAED_DAG')
                        results.append(('VAED_DAG', 'XGBoost', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(VAED_DAG)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_sel, ytr, Xva_sel, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_VAED_DAG')
                        results.append(('VAED_DAG', 'MLP', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(VAED_DAG)训练失败：{e}', file=sys.stderr)
            except Exception as e:
                print(f'VAED_DAG 下游(LGB/XGB/MLP)失败：{e}', file=sys.stderr)
            try:
                if getattr(args, 'gf_enabled', False):
                    A_core = A_vaed_core
                    if (args.fs_adjacency_mode if args.fs_adjacency_mode else args.adjacency_mode) in ('skeleton', 'semi'):
                        A_core = (A_core + A_core.T > 0).astype(int)
                    GF_tr = _graph_poly_features(np, Xtr, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                    GF_va = _graph_poly_features(np, Xva, A_core, orders=list(map(int, args.gf_orders)), mode=str(args.gf_mode))
                    try:
                        graph_feat_counts['VAED_DAG_GF'] = int(GF_tr.shape[1])
                    except Exception:
                        pass
                    if args.gf_attach == 'concat':
                        Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                        Xva_gf = np.concatenate([Xva, GF_va], axis=1)
                    else:
                        Xtr_gf, Xva_gf = (GF_tr, GF_va)
                    try:
                        _, _, mtr, mva = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_VAED_DAG_GF')
                        results.append(('VAED_DAG_GF', 'LightGBM', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'LightGBM(VAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, 'soc_VAED_DAG_GF')
                        results.append(('VAED_DAG_GF', 'XGBoost', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'XGBoost(VAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_VAED_DAG_GF')
                        results.append(('VAED_DAG_GF', 'MLP', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'MLP(VAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'GraphSAGE', A_vaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_VAED_DAG_GF')
                        results.append(('VAED_DAG_GF', 'GraphSAGE', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'GraphSAGE(VAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'GCN', A_vaed, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, 'soc_VAED_DAG_GF')
                        results.append(('VAED_DAG_GF', 'GCN', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'GCN(VAED_DAG_GF)训练失败：{e}', file=sys.stderr)
                    try:
                        _, _, mtr, mva = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'ECMPNN', A_vaed, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, 'soc_VAED_DAG_GF')
                        results.append(('VAED_DAG_GF', 'ECMPNN', edge_counts.get('VAED_DAG', None), *mtr, *mva))
                    except Exception as e:
                        print(f'ECMPNN(VAED_DAG_GF)训练失败：{e}', file=sys.stderr)
            except Exception as e:
                print(f'VAED 图滤波特征失败：{e}', file=sys.stderr)
        except Exception as e:
            print(f'VAED+分簇DAG流程失败：{e}', file=sys.stderr)
    try:
        _, _, mtr, mva = train_and_eval_rf(np, Xtr, ytr, Xva, yva, args.outdir, 'soc_RF')
        results.append(('NONE', 'RandomForest', None, *mtr, *mva))
    except Exception as e:
        print(f'随机森林训练失败：{e}', file=sys.stderr)
    try:
        _, _, mtr, mva = train_and_eval_lgb(np, Xtr, ytr, Xva, yva, args.outdir, 'soc_LGB')
        results.append(('NONE', 'LightGBM', None, *mtr, *mva))
    except Exception as e:
        print(f'LightGBM 训练失败：{e}', file=sys.stderr)
    try:
        _, _, mtr, mva = train_and_eval_xgb(np, Xtr, ytr, Xva, yva, args.outdir, 'soc_XGB')
        results.append(('NONE', 'XGBoost', None, *mtr, *mva))
    except Exception as e:
        print(f'XGBoost 训练失败：{e}', file=sys.stderr)
    try:
        _, _, mtr, mva = train_and_eval_mlp(torch, np, Xtr, ytr, Xva, yva, args.hidden, max(50, args.sage_epochs), args.lr, args.outdir, 'soc_MLP')
        results.append(('NONE', 'MLP', None, *mtr, *mva))
    except Exception as e:
        print(f'MLP 训练失败：{e}', file=sys.stderr)
    try:
        import pandas as pd
        if results:
            cols = ['Adjacency', 'Model', 'Edges', 'Train_MSE', 'Train_MAE', 'Train_R2', 'Valid_MSE', 'Valid_MAE', 'Valid_R2']
            df_sum = pd.DataFrame(results, columns=cols)
            try:
                df_sum['GraphFeatures'] = df_sum['Adjacency'].map(lambda k: graph_feat_counts.get(k, 0)).astype(int)
                df_sum = df_sum[['Adjacency', 'Model', 'Edges', 'GraphFeatures', 'Train_MSE', 'Train_MAE', 'Train_R2', 'Valid_MSE', 'Valid_MAE', 'Valid_R2']]
            except Exception:
                pass
            df_sum.to_csv(os.path.join(args.outdir, 'metrics_summary.csv'), index=False, encoding='utf-8-sig')
    except Exception:
        pass
    if results and any((r[0] == 'FCI' and r[1] == 'GraphSAGE' for r in results)):
        try:
            model = GraphSAGE(Xtr.shape[1], args.hidden, 2, target_idx, torch.tensor(adj_fci, dtype=torch.float32, device=torch.device('cpu')))
            model.to(torch.device('cpu'))
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = torch.nn.MSELoss()
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=torch.device('cpu'))
            ytr_t = torch.tensor(ytr, dtype=torch.float32, device=torch.device('cpu'))
            for ep in range(args.sage_epochs):
                model.train()
                opt.zero_grad()
                pred = model(Xtr_t)
                loss = loss_fn(pred, ytr_t)
                loss.backward()
                opt.step()
            df_test = df.loc[val_idx]
            predictions = predict_test_sites(torch, np, df_test, var_cols, args.target_col, args.time_col, args.group_col, model, target_idx, meanX, stdX, meany, stdy, encoders)
            pd.DataFrame(predictions).to_csv(os.path.join(args.outdir, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
            print(f"测试预测已保存到 {os.path.join(args.outdir, 'test_predictions.csv')}")
        except Exception as e:
            print(f'测试预测失败：{e}', file=sys.stderr)
    print(f'完成。输出目录：{args.outdir}')
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))