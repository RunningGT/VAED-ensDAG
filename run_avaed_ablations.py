import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.simplefilter('ignore')

# 配置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Import functions from experiment.py
from experiment import (
    select_variables, build_site_split_indices, label_encode_dataframe,
    apply_encoders, zscore, _vaed_train_and_cluster, _compute_attention_responsibilities,
    _aggregate_cluster_dags, make_gnn_adj, train_and_eval_lgb, train_and_eval_xgb,
    train_and_eval_mlp, train_and_eval_model, GraphSAGE, GCN, ECMPNN, select_features_by_adj,
    _graph_poly_features
)

class MockArgs:
    def __init__(self):
        # 默认参数设定
        self.data = "data.xlsx"
        self.sheet = 0
        self.group_col = "Ecological Station Code"
        self.time_col = "carbon_time"
        self.target_col = "Soil Organic Carbon Density (kg/m2)"
        self.train_sites = 17
        self.test_sites = 2
        self.split_seed = 42
        
        self.vaed_epochs = 500
        self.vaed_lr = 1e-3
        self.vaed_hidden = 64
        self.vaed_latent_dim = 16
        self.vaed_lambda3 = 0.1
        self.vaed_gmm_update = 10
        self.vaed_agg_threshold = 0.1
        self.avaed_agg_threshold = 0.1
        
        self.dag_epochs = 500
        self.dag_hidden = 64
        self.dag_lr = 5e-3
        self.dag_lambda_acyc = 1.0
        self.dag_lambda_sparse = 1e-4
        self.dag_threshold = 0.2
        
        self.hidden = 64
        self.sage_epochs = 500
        self.ecmpnn_epochs = 500
        self.lr = 1e-3
        
        self.adjacency_mode = "directed"
        self.fs_adjacency_mode = "semi"
        self.augment_target_k = 0
        self.symmetrize_adj = True
        self.gf_enabled = True
        self.gf_orders = [1, 2]
        self.gf_mode = "both"
        self.gf_attach = "concat"
        self.rand_seed = 84
        self.agg_target_edges = 0
        
        self.outdir = "outputs_2"

def _plot_latent_scatter(mu, resp, labels, K, outdir):
    """提取Z的散点图（按簇或按站点着色）"""
    from sklearn.decomposition import PCA
    if mu.shape[1] > 2:
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(mu)
    else:
        mu_2d = mu
        
    cluster_assign = resp.argmax(axis=1)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=cluster_assign, cmap='tab10', alpha=0.7, s=15)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'GMM Latent Space (K={K}) - by Cluster')
    
    plt.subplot(1, 2, 2)
    # Convert string labels to factors for coloring
    uniq_labels = list(set(labels))
    label_map = {l: i for i, l in enumerate(uniq_labels)}
    c_labels = [label_map[l] for l in labels]
    
    scatter2 = plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=c_labels, cmap='tab20', alpha=0.7, s=15)
    plt.title(f'GMM Latent Space (K={K}) - by Station')
    
    plt.tight_layout()
    out_path = os.path.join(outdir, f'gmm_latent_scatter_K{K}.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[{K}] Latent scatter plot saved to {out_path}")

def main():
    args = MockArgs()
    os.makedirs(args.outdir, exist_ok=True)
    
    print("正在读取数据...")
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    
    # 匿名化处理：不能泄露station的名称，plot code被标记为unkown
    # 假设 '生态站代码' 或 'Ecological Station Code' 存在
    group_candidates = ['Ecological Station Code', '生态站代码', 'station', 'Station']
    for gc in group_candidates:
        if gc in df.columns:
            args.group_col = gc
            break
            
    plot_candidates = ['样地代码', 'plot_id', 'Plot']
    for pc in plot_candidates:
        if pc in df.columns:
            df[pc] = 'unknown'
            
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
    yva = (yva.reshape(-1, 1) - meany) / (stdy + 1e-8)
    yva = yva.reshape(-1)
    
    station_labels = tr_df_full[args.group_col].loc[train_idx_eff].to_numpy()
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.metrics.cluster import normalized_mutual_info_score
    
    k_values = [2, 3, 4, 5, 6]
    ablation_results = []
    clustering_metrics_list = []
    
    for K in k_values:
        print(f"\n==================== 开始测试 K = {K} ====================")
        args.vaed_clusters = K
        
        # 对于 K=3 可能出现后验塌缩(Posterior Collapse)，尝试使用不同的 seed 去打破局部最优
        current_seed = args.split_seed if K != 3 else args.split_seed + 100
        
        vaed_res = _vaed_train_and_cluster(
            torch, np, Xtr,
            K=K, z_dim=args.vaed_latent_dim, hidden=args.vaed_hidden,
            epochs=args.vaed_epochs, lr=args.vaed_lr, lambda3=args.vaed_lambda3,
            gmm_update_every=args.vaed_gmm_update, seed=current_seed,
            station_labels=station_labels, outdir=args.outdir
        )
        
        _plot_latent_scatter(vaed_res['mu'], vaed_res['resp'], station_labels, K, args.outdir)
        
        # 保存真实的潜变量和分配概率供绘图使用
        pd.DataFrame(vaed_res['mu']).to_csv(os.path.join(args.outdir, f'avaed_K{K}_latent_mu.csv'), index=False)
        pd.DataFrame(vaed_res['resp']).to_csv(os.path.join(args.outdir, f'avaed_K{K}_latent_resp.csv'), index=False)
        pd.DataFrame(station_labels, columns=["Station"]).to_csv(os.path.join(args.outdir, f'avaed_K{K}_station_labels.csv'), index=False)
        
        attn = _compute_attention_responsibilities(np, vaed_res['mu'], vaed_res['gmm'], gamma=0.5, beta=1.0, temp_fine=1.0, temp_coarse=2.0)
        
        dag_runner_ctx = {"cat_cols": list(encoders.keys())}
        A_avaed, _, _ = _aggregate_cluster_dags(
            np, var_cols, work_tr, attn, args, dag_runner_ctx,
            agg_threshold=args.avaed_agg_threshold, prefix=f'avaed_K{K}'
        )
        
        A_avaed_core = A_avaed.copy()
        A_avaed_gnn = make_gnn_adj(A_avaed_core, symmetrize=args.symmetrize_adj, add_self_loop=True)
        num_edges = int(np.count_nonzero(A_avaed_core))
        
        # 计算聚类指标
        mu_tr = vaed_res['mu']
        cluster_assign = vaed_res['resp'].argmax(axis=1)
        
        # 防止簇数为1时的计算错误
        if len(set(cluster_assign)) > 1:
            sil_score = silhouette_score(mu_tr, cluster_assign)
            dbi_score = davies_bouldin_score(mu_tr, cluster_assign)
            chi_score = calinski_harabasz_score(mu_tr, cluster_assign)
            nmi_score = normalized_mutual_info_score(station_labels, cluster_assign)
        else:
            sil_score, dbi_score, chi_score, nmi_score = None, None, None, None
            
        print(f"[K={K}] 聚类指标 -> Silhouette: {sil_score}, DBI: {dbi_score}, CH: {chi_score}, NMI: {nmi_score}")
        clustering_metrics_list.append({
            'K': K, 
            'Edges': num_edges,
            'Silhouette': sil_score, 
            'DBI': dbi_score, 
            'CHI': chi_score, 
            'NMI (vs Stations)': nmi_score
        })
        
        try:
            # 应用图形滤波器的逻辑
            if args.fs_adjacency_mode in ('skeleton', 'semi'):
                # 针对骨架图或半有向图模式，构建无向图
                A_core = ((A_avaed_core + A_avaed_core.T) > 0).astype(int)
            else:
                A_core = A_avaed_core

            GF_tr = _graph_poly_features(np, Xtr, A_core, orders=args.gf_orders, mode=args.gf_mode)
            GF_va = _graph_poly_features(np, Xva, A_core, orders=args.gf_orders, mode=args.gf_mode)

            if args.gf_attach == 'concat':
                Xtr_gf = np.concatenate([Xtr, GF_tr], axis=1)
                Xva_gf = np.concatenate([Xva, GF_va], axis=1)
            else:
                Xtr_gf, Xva_gf = GF_tr, GF_va

            print(f"[K={K}] 提取 Graph Filters 特征维度: {GF_tr.shape[1]}")
            
            print(f"[K={K}] 正在训练 LightGBM (GF)...")
            _, _, _, mva_lgb = train_and_eval_lgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, f'AVAED_K{K}_GF_LGB')
            ablation_results.append({'K': K, 'Model': 'LightGBM_GF', 'GraphFeatures': GF_tr.shape[1], 'MSE': mva_lgb[0], 'MAE': mva_lgb[1], 'R2': mva_lgb[2]})
            
            print(f"[K={K}] 正在训练 XGBoost (GF)...")
            _, _, _, mva_xgb = train_and_eval_xgb(np, Xtr_gf, ytr, Xva_gf, yva, args.outdir, f'AVAED_K{K}_GF_XGB')
            ablation_results.append({'K': K, 'Model': 'XGBoost_GF', 'GraphFeatures': GF_tr.shape[1], 'MSE': mva_xgb[0], 'MAE': mva_xgb[1], 'R2': mva_xgb[2]})
            
            print(f"[K={K}] 正在训练 MLP (GF)...")
            _, _, _, mva_mlp = train_and_eval_mlp(torch, np, Xtr_gf, ytr, Xva_gf, yva, args.hidden, args.sage_epochs, args.lr, args.outdir, f'AVAED_K{K}_GF_MLP')
            ablation_results.append({'K': K, 'Model': 'MLP_GF', 'GraphFeatures': GF_tr.shape[1], 'MSE': mva_mlp[0], 'MAE': mva_mlp[1], 'R2': mva_mlp[2]})
            
            print(f"[K={K}] 正在训练 GraphSAGE (GF)...")
            _, _, _, mva_sage = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'GraphSAGE', A_avaed_gnn, args.hidden, args.sage_epochs, args.lr, target_idx, args.outdir, f'AVAED_K{K}_GF_SAGE')
            ablation_results.append({'K': K, 'Model': 'GraphSAGE_GF', 'GraphFeatures': GF_tr.shape[1], 'MSE': mva_sage[0], 'MAE': mva_sage[1], 'R2': mva_sage[2]})
            
            print(f"[K={K}] 正在训练 ECMPNN (GF)...")
            _, _, _, mva_ecmpnn = train_and_eval_model(torch, np, Xtr_gf, ytr, Xva_gf, yva, 'ECMPNN', A_avaed_gnn, args.hidden, args.ecmpnn_epochs, args.lr, target_idx, args.outdir, f'AVAED_K{K}_GF_ECMPNN')
            ablation_results.append({'K': K, 'Model': 'ECMPNN_GF', 'GraphFeatures': GF_tr.shape[1], 'MSE': mva_ecmpnn[0], 'MAE': mva_ecmpnn[1], 'R2': mva_ecmpnn[2]})

        except Exception as e:
            print(f"[K={K}] 模型评估失败：{e}")
            
    df_res = pd.DataFrame(ablation_results)
    res_path = os.path.join(args.outdir, 'avaed_k_ablation_metrics.csv')
    df_res.to_csv(res_path, index=False, encoding='utf-8-sig')
    
    df_clus = pd.DataFrame(clustering_metrics_list)
    clus_path = os.path.join(args.outdir, 'avaed_k_cluster_metrics.csv')
    df_clus.to_csv(clus_path, index=False, encoding='utf-8-sig')
    print(f"聚类特性评估结果保存在 {clus_path}")
    print(f"\n全部测试完成！结果保存在 {res_path}")

if __name__ == '__main__':
    main()
