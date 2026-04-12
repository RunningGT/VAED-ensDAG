import argparse
import os
import sys
from typing import List, Dict, Tuple, Optional
from typing import Any
try:
    import torch
except Exception:
    pass

def ensure_imports():
    try:
        import pandas as pd
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib
        import torch
        import warnings
        warnings.simplefilter('ignore')
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return (pd, np, nx, plt, torch)
    except Exception as e:
        print('需要 pandas/numpy/networkx/matplotlib/torch，请安装后重试：\npip install pandas numpy networkx matplotlib torch', file=sys.stderr)
        raise

def try_import_causallearn():
    try:
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.utils.cit import fisherz
        try:
            from causallearn.search.ConstraintBased.PC import pc
        except Exception:
            pc = None
        return (fci, fisherz, pc)
    except Exception as e:
        return (None, None, None)

def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description='按样地与时间划分训练/验证，FCI因果发现+GraphSAGE/ECMPNN预测SOC')
    p.add_argument('--data', required=True, help='输入数据文件 data.xlsx')
    p.add_argument('--sheet', default=0, help='工作表名或索引，默认第一个')
    p.add_argument('--group-col', default='Ecological Station Code', help='按该列分组进行时间切分，默认 样地代码')
    p.add_argument('--time-col', default='carbon_time', help='时间列，默认 carbon_time (Excel序列日期或数值)')
    p.add_argument('--target-col', default='Soil Organic Carbon Density (kg/m2)', help='目标列，默认 土壤有机碳密度（kg/m2）')
    p.add_argument('--split-ratio', type=float, default=0.7, help='每个样地内按时间排序前比例作为训练，默认0.7')
    p.add_argument('--train-sites', type=int, default=17, help='训练样地个数，默认40')
    p.add_argument('--test-sites', type=int, default=2, help='测试样地个数，默认20')
    p.add_argument('--split-seed', type=int, default=42, help='用于站点划分（train/test 样地选择）的随机种子，默认42')
    p.add_argument('--alpha', type=float, default=0.05, help='FCI 显著性水平，默认0.05')
    p.add_argument('--graph-png', default='fci_graph.png', help='保存因果图的PNG路径')
    p.add_argument('--adj-out', default='fci_adjacency.csv', help='保存邻接矩阵CSV')
    p.add_argument('--sage-epochs', type=int, default=500, help='GraphSAGE训练轮数，默认200')
    p.add_argument('--ecmpnn-epochs', type=int, default=500, help='ECMPNN训练轮数，默认200')
    p.add_argument('--lr', type=float, default=0.001, help='学习率，默认1e-2')
    p.add_argument('--hidden', type=int, default=64, help='隐藏维度，默认32')
    p.add_argument('--outdir', default='outputs_causal_gnn', help='输出目录')
    p.add_argument('--dag-epochs', type=int, default=500, help='DAG-GNN 训练轮数，默认200')
    p.add_argument('--dag-hidden', type=int, default=64, help='DAG-GNN 解码器隐藏维度，默认32')
    p.add_argument('--dag-lr', type=float, default=0.001, help='DAG-GNN 学习率，默认5e-3')
    p.add_argument('--dag-lambda-acyc', type=float, default=1.0, help='DAG acyclicity 约束权重，默认1.0')
    p.add_argument('--dag-lambda-sparse', type=float, default=0.0001, help='DAG 稀疏正则权重，默认1e-4')
    p.add_argument('--dag-threshold', type=float, default=0.2, help='将连续邻接转换为0/1的阈值，默认0.3')
    p.add_argument('--dag-graph-png', default='dag_gnn_graph.png', help='DAG-GNN 学习到的图PNG')
    p.add_argument('--dag-adj-out', default='dag_gnn_adjacency.csv', help='DAG-GNN 邻接矩阵CSV')
    p.add_argument('--dag-edges-out', default='dag_gnn_edges.txt', help='DAG-GNN 边调试信息')
    p.add_argument('--augment-target-k', type=int, default=5, help='按与目标的相关性增强到目标的入边数量，默认5；0表示不增强')
    p.add_argument('--symmetrize-adj', action='store_true', default=False, help='对邻接矩阵做对称化；默认不对称以保留方向性')
    p.add_argument('--adjacency-mode', choices=['directed', 'semi', 'skeleton'], default='directed', help='用于GNN的邻接类型：directed=仅确定方向的边；semi=确定方向按方向，不确定的边双向；skeleton=无向骨架(双向)。默认directed')
    p.add_argument('--fs-adjacency-mode', choices=['directed', 'semi', 'skeleton'], default=None, help='用于表格模型特征选择的邻接类型；不指定时默认沿用 --adjacency-mode')
    p.add_argument('--rand-edges', type=int, default=10, help='随机图对照使用的骨架边数（按无向对计数）。0 表示退回使用FCI的边数')
    p.add_argument('--rand-seed', type=int, default=42, help='随机图对照的随机种子')
    p.add_argument('--vaed-enabled', action='store_true', default=False, help='启用VAED+分簇DAG上游实验组（保留原DAG-GNN 组）')
    p.add_argument('--vaed-clusters', type=int, default=5, help='VAED聚类簇数 K')
    p.add_argument('--vaed-latent-dim', type=int, default=16, help='VAED潜变量维度')
    p.add_argument('--vaed-hidden', type=int, default=64, help='VAED编码器/解码器隐藏维度')
    p.add_argument('--vaed-epochs', type=int, default=500, help='VAED总训练轮数')
    p.add_argument('--vaed-lr', type=float, default=0.001, help='VAED学习率')
    p.add_argument('--vaed-lambda3', type=float, default=0.1, help='VAED中距离项权重 λ3')
    p.add_argument('--vaed-gmm-update', type=int, default=10, help='每多少个epoch更新一次GMM')
    p.add_argument('--vaed-agg-threshold', type=float, default=0.2, help='VAED 聚合簇内DAG的阈值（按簇权重比例≥该值则保留边）')
    p.add_argument('--avaed-agg-threshold', type=float, default=0.2, help='AVAED 注意力聚合阈值（默认更低以避免空图）')
    p.add_argument('--agg-target-edges', type=int, default=0, help='VAED/AVAED 聚合图目标边数；>0 时自动按绝对值阈值匹配该边数')
    p.add_argument('--vaed-pretrain', type=int, default=50, help='VAED仅VAE预训练轮数（不启用聚类距离项）')
    p.add_argument('--vaed-alpha0', type=float, default=1.0, help='GMM先验平滑项（Dirichlet-like），避免簇塌缩')
    p.add_argument('--vaed-tau-max', type=float, default=100.0, help='GMM精度上限，避免数值爆炸')
    p.add_argument('--vaed-sigma2-min', type=float, default=0.001, help='VAE后验方差下限，避免数值不稳')
    p.add_argument('--ivae-enabled', action='store_true', default=False, help='启用 iVAE 实验组：U(默认样地代码) 指导潜变量 Z，并学习 Z 的因果图')
    p.add_argument('--ivae-u-type', choices=['station', 'plot', 'vaed', 'avaed'], default='plot', help='iVAE 的辅助变量 U 取值：station=生态站代码，plot=样地代码（默认），vaed=用VAED簇，avaed=用注意力簇')
    p.add_argument('--ivae-latent-dim', type=int, default=16, help='iVAE 潜变量维度')
    p.add_argument('--ivae-embedding', type=int, default=16, help='iVAE 辅助变量 U 的嵌入维度')
    p.add_argument('--ivae-hidden', type=int, nargs='+', default=[64, 32], help='iVAE 编码器/解码器隐藏层维度列表')
    p.add_argument('--ivae-lr', type=float, default=0.0005, help='iVAE 学习率')
    p.add_argument('--ivae-lambda-dag', type=float, default=0.5, help='iVAE DAG 约束乘子初值')
    p.add_argument('--ivae-lambda-sparse', type=float, default=0.0008, help='iVAE L1 稀疏正则')
    p.add_argument('--ivae-c-init', type=float, default=1.0, help='iVAE 增广拉格朗日惩罚系数初值 c')
    p.add_argument('--ivae-rho', type=float, default=1.1, help='iVAE 增广拉格朗日惩罚系数增长率 rho')
    p.add_argument('--ivae-h-tol', type=float, default=1e-08, help='iVAE DAG 约束容忍度 h_tol')
    p.add_argument('--ivae-al-iters', type=int, default=500, help='iVAE 增广拉格朗日外层迭代次数')
    p.add_argument('--ivae-kl-warmup-iters', type=int, default=20, help='iVAE KL warm-up 迭代数')
    p.add_argument('--ivae-kl-start', type=float, default=0.0, help='iVAE KL warm-up 起始系数')
    p.add_argument('--ivae-kl-end', type=float, default=1.0, help='iVAE KL warm-up 终止系数')
    p.add_argument('--ivae-batch-size', type=int, default=16, help='iVAE 训练批大小')
    p.add_argument('--ivae-graph-threshold', type=float, default=0.05, help='iVAE 因果图可视化/计数基础阈值（按绝对值），可与 --ivae-target-edges 配合')
    p.add_argument('--ivae-target-edges', type=int, default=0, help='iVAE 图目标边数；>0 时自动按绝对值阈值匹配该边数')
    p.add_argument('--ivae-graph-png', default='ivae_graph.png', help='iVAE 学习到的 Z 因果图PNG')
    p.add_argument('--ivae-adj-out', default='ivae_adjacency.csv', help='iVAE 邻接矩阵CSV（连续权重会另存 *_continuous.csv）')
    p.add_argument('--gf-enabled', action='store_true', default=True, help='启用图多项式滤波+池化的全局特征，并用于表格模型')
    p.add_argument('--gf-orders', type=int, nargs='+', default=[1, 2], help='图多项式的阶数列表，例如 1 2 3')
    p.add_argument('--gf-mode', choices=['A', 'AT', 'sym', 'both'], default='both', help='滤波方向：A(出边)、AT(入边)、sym(对称)、both(合并A与AT)')
    p.add_argument('--gf-attach', choices=['concat', 'only'], default='concat', help='与原始X拼接(concat)或仅使用GF(only)')
    return p.parse_args(argv)

def build_site_split_indices(pd, np, df, group_col: str, train_sites: int, test_sites: int, split_seed: int=42) -> Tuple[List[int], List[int]]:
    df_valid = df[~df[group_col].isna()].copy()
    unique_sites = df_valid[group_col].dropna().unique()
    total_sites = len(unique_sites)
    try:
        np.random.seed(int(split_seed))
    except Exception:
        np.random.seed(42)
    np.random.shuffle(unique_sites)
    if total_sites == 0:
        return ([], [])
    train_eff = min(train_sites, max(1, total_sites - 1))
    test_eff = min(test_sites, total_sites - train_eff)
    if test_eff <= 0:
        test_eff = 1
        train_eff = max(1, total_sites - test_eff)
    train_sites_list = unique_sites[:train_eff]
    test_sites_list = unique_sites[train_eff:train_eff + test_eff]
    train_idx = df_valid[df_valid[group_col].isin(train_sites_list)].index.tolist()
    val_idx = df_valid[df_valid[group_col].isin(test_sites_list)].index.tolist()
    print(f'按{group_col}划分：可用站点数={total_sites}，训练站点数={len(train_sites_list)}，测试站点数={len(test_sites_list)}，split_seed={split_seed}')
    return (train_idx, val_idx)

def select_variables(df, target_col: str, exclude_cols: List[str]) -> List[str]:
    vars_all = []
    for c in df.columns:
        if c == target_col:
            vars_all.append(c)
        elif c not in exclude_cols:
            vars_all.append(c)
    return vars_all

def label_encode_dataframe(pd, df, exclude_numeric: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
    encoders: Dict[str, Dict] = {}
    numeric_cols = []
    for c in df.columns:
        s = df[c]
        sc = pd.to_numeric(s, errors='coerce')
        if sc.notna().mean() > 0.9:
            df[c] = sc.fillna(0)
            numeric_cols.append(c)
        else:
            cats = s.astype(str).fillna('')
            uniq = sorted(cats.unique())
            enc = {v: i for i, v in enumerate(uniq)}
            df[c] = cats.map(enc).fillna(0).astype(int)
            encoders[c] = enc
            numeric_cols.append(c)
    return (encoders, numeric_cols)

def zscore(pd, np, arr, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(arr, axis=0)
    if std is None:
        std = np.nanstd(arr, axis=0) + 1e-08
    return ((arr - mean) / std, mean, std)

def make_gnn_adj(adj, symmetrize: bool=True, add_self_loop: bool=True):
    import numpy as _np
    A = adj.astype(float).copy()
    if symmetrize:
        A = (A + A.T > 0).astype(float)
    if add_self_loop:
        _np.fill_diagonal(A, 1.0)
    else:
        _np.fill_diagonal(A, 0.0)
    return A

def compute_missing_zero_stats(pd, np, df, cols):
    rows = []
    sub = df[cols]
    for c in cols:
        s = sub[c]
        sn = pd.to_numeric(s, errors='coerce')
        n = len(s)
        nan_cnt = int(s.isna().sum())
        zero_cnt = int((sn == 0).sum(skipna=True))
        comb_cnt = int((s.isna() | (sn == 0)).sum(skipna=True))
        rows.append({'column': c, 'n_rows': n, 'nan_count': nan_cnt, 'zero_count': zero_cnt, 'nan_or_zero_count': comb_cnt, 'nan_ratio': nan_cnt / n if n else 0.0, 'zero_ratio': zero_cnt / n if n else 0.0, 'combined_ratio': comb_cnt / n if n else 0.0})
    return pd.DataFrame(rows)

def augment_adj_with_target(np, adj, X_tr, y_tr, target_idx: int, k: int):
    """
    基于训练集的皮尔逊相关性，选取与目标最相关的K个特征，强制添加特征->目标 的入边。
    - adj: np.ndarray [N,N]，会就地或拷贝后修改（返回修改后的邻接）。
    - X_tr: 训练特征矩阵，包含目标列（但通常我们会在外部传入目标列仍为原始值的副本以做相关性）。
    - y_tr: 训练目标向量（原始值）。
    - 对于特征j，相关性以存在非零的样本为主（避免把0当成真实值计算）。
    """
    A = adj.copy()
    N = A.shape[0]
    if k <= 0 or N != X_tr.shape[1]:
        return A
    cors = []
    for j in range(N):
        if j == target_idx:
            continue
        x = X_tr[:, j]
        m = x != 0
        if m.sum() < 5:
            c = 0.0
        else:
            try:
                xv = x[m]
                yv = y_tr[m]
                if np.std(xv) < 1e-08 or np.std(yv) < 1e-08:
                    c = 0.0
                else:
                    c = float(np.corrcoef(xv, yv)[0, 1])
            except Exception:
                c = 0.0
        cors.append((j, abs(c)))
    cors.sort(key=lambda t: t[1], reverse=True)
    for j, _ in cors[:k]:
        A[j, target_idx] = 1
    return A

def select_features_by_adj(np, adj, target_idx: int, mode: str='nbr', fallback_to_all: bool=False):
    """
    基于邻接选择与目标节点相关的特征列索引。
    """
    try:
        N = adj.shape[0]
        all_idx = [i for i in range(N) if i != target_idx]
        print(f'[DEBUG] 特征选择开始: N={N}, target_idx={target_idx}, mode={mode}, fallback_to_all={fallback_to_all}')
        print(f'[DEBUG] 邻接矩阵形状: {adj.shape}, 非零元素: {np.count_nonzero(adj)}')
        if mode == 'in':
            idx = [j for j in range(N) if j != target_idx and abs(float(adj[j, target_idx])) > 0]
            print(f"[DEBUG] 'in'模式: 找到 {len(idx)} 个指向目标的特征")
        elif mode == 'out':
            idx = [j for j in range(N) if j != target_idx and abs(float(adj[target_idx, j])) > 0]
            print(f"[DEBUG] 'out'模式: 找到 {len(idx)} 个目标指向的特征")
        else:
            in_set = {j for j in range(N) if j != target_idx and abs(float(adj[j, target_idx])) > 0}
            out_set = {j for j in range(N) if j != target_idx and abs(float(adj[target_idx, j])) > 0}
            idx = sorted(list(in_set | out_set))
            print(f"[DEBUG] 'nbr'模式: in_set={len(in_set)}, out_set={len(out_set)}, 合并后={len(idx)}")
        if idx:
            print(f'[DEBUG] 选中的特征索引: {idx}')
        else:
            print(f'[DEBUG] 没有选中任何特征!')
        if not idx:
            if fallback_to_all:
                print(f'⚠️ 警告: 邻接矩阵未选择任何特征，退回使用全部 {len(all_idx)} 个特征')
                return all_idx
            else:
                print(f'⚠️ 警告: 邻接矩阵未选择任何特征，返回空特征集')
                return []
        print(f'[DEBUG] 最终返回 {len(idx)} 个特征')
        return idx
    except Exception as e:
        print(f'[DEBUG] 特征选择异常: {e}')
        try:
            N = adj.shape[0]
            fallback_idx = [i for i in range(N) if i != target_idx]
            print(f'[DEBUG] 异常回退: 使用全部 {len(fallback_idx)} 个特征')
            return fallback_idx
        except Exception:
            print(f'[DEBUG] 严重异常: 返回空列表')
            return []

def apply_encoders(pd, df, encoders: Dict[str, Dict]):
    out = df.copy()
    for col in out.columns:
        if col in encoders:
            enc = encoders[col]
            if isinstance(enc, dict) and len(enc) > 0:
                unk_id = max(enc.values()) + 1
                out[col] = out[col].astype(str).fillna('').apply(lambda v: enc.get(v, unk_id)).astype(int)
            else:
                out[col] = out[col].astype(str).fillna('')
        else:
            out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)
    return out

def build_endpoint_matrices(var_cols: List[str], edges_idx: List[Tuple[int, int, str]]):
    """
    - T[i,j] = 1 表示在边(i,j)上 i 端为 Tail('-')
    - H[i,j] = 1 表示在边(i,j)上 i 端为 Arrow('>')
    - C[i,j] = 1 表示在边(i,j)上 i 端为 Circle('o')
    """
    import numpy as _np
    N = len(var_cols)
    T = _np.zeros((N, N), dtype=int)
    H = _np.zeros((N, N), dtype=int)
    C = _np.zeros((N, N), dtype=int)
    for i, j, pat in edges_idx:
        if i >= N or j >= N or i == j:
            continue
        if pat in ('->',):
            T[i, j] = 1
            H[j, i] = 1
        elif pat in ('<-',):
            T[j, i] = 1
            H[i, j] = 1
        elif pat in ('<->',):
            H[i, j] = 1
            H[j, i] = 1
        elif pat in ('o->',):
            C[i, j] = 1
            H[j, i] = 1
        elif pat in ('<-o',):
            H[i, j] = 1
            C[j, i] = 1
        elif pat in ('o-o',):
            C[i, j] = 1
            C[j, i] = 1
        else:
            T[i, j] = 1
            T[j, i] = 1
    return (T, H, C)

def endpoints_to_directed_adj(np, T, H):
    N = T.shape[0]
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if T[i, j] == 1 and H[j, i] == 1:
                A[i, j] = 1
    return A

def endpoints_to_skeleton(np, T, H, C):
    # 忽略方向的骨架邻接：任意端点存在即认为连接存在，返回对称邻接。
    N = T.shape[0]
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            present = T[i, j] or H[i, j] or C[i, j] or T[j, i] or H[j, i] or C[j, i]
            if present:
                A[i, j] = 1
                A[j, i] = 1
    return A

def endpoints_to_semi_directed(np, T, H, C):
    # 确定方向的用 i->j；未定向或含圆圈的不确定边以双向连接。
    A_dir = endpoints_to_directed_adj(np, T, H)
    A_skel = endpoints_to_skeleton(np, T, H, C)
    A = A_dir.copy()
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if A_skel[i, j] == 1 and A[i, j] == 0 and (A[j, i] == 0):
                A[i, j] = 1
                A[j, i] = 1
    return A

def random_endpoint_graph(np, N: int, m_edges: int, seed: int=42):
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng.shuffle(pairs)
    pairs = pairs[:max(0, min(m_edges, len(pairs)))]
    import numpy as _np
    T = _np.zeros((N, N), dtype=int)
    H = _np.zeros((N, N), dtype=int)
    C = _np.zeros((N, N), dtype=int)
    for i, j in pairs:
        r = rng.random()
        if r < 1 / 3:
            T[i, j] = 1
            H[j, i] = 1
        elif r < 2 / 3:
            T[j, i] = 1
            H[i, j] = 1
        elif r < 5 / 6:
            H[i, j] = 1
            H[j, i] = 1
        else:
            r2 = rng.random()
            if r2 < 1 / 3:
                C[i, j] = 1
                C[j, i] = 1
            elif r2 < 2 / 3:
                C[i, j] = 1
                H[j, i] = 1
            else:
                H[i, j] = 1
                C[j, i] = 1
    A_dir = endpoints_to_directed_adj(np, T, H)
    A_skel = endpoints_to_skeleton(np, T, H, C)
    A_semi = endpoints_to_semi_directed(np, T, H, C)
    return {'tail': T, 'arrow': H, 'circle': C, 'adj': A_dir, 'skeleton': A_skel, 'semi': A_semi}

def _parse_edges_generic(cg, var_cols: Optional[List[str]]=None) -> List[Tuple[int, int, str]]:
    """尝试从 causallearn 的结果对象中解析边。
    返回列表[(i,j,pat)]，其中 pat 描述方向模式字符串，便于调试：'->','<-','<->','---','o->','<-o','o-o','?'.
    """
    import re
    edges_raw = []
    edge_sources = []
    try:
        edge_sources = list(cg.get_graph_edges())
        for e in edge_sources:
            s = str(e)
            ms = re.findall('X(\\d+)', s)
            if len(ms) >= 2:
                i = max(0, int(ms[0]) - 1)
                j = max(0, int(ms[1]) - 1)
                pat = '---'
                if '<->' in s:
                    pat = '<->'
                elif '-->' in s or '->' in s:
                    pat = '->'
                elif '<--' in s or '<-' in s:
                    pat = '<-'
                elif 'o->' in s:
                    pat = 'o->'
                elif '<-o' in s:
                    pat = '<-o'
                elif 'o-o' in s:
                    pat = 'o-o'
                edges_raw.append((i, j, pat))
        if edges_raw:
            return edges_raw
    except Exception:
        pass
    try:
        nxg = None
        if hasattr(cg, 'to_nx_graph'):
            nxg = cg.to_nx_graph()
        elif hasattr(cg, 'nx_graph'):
            nxg = getattr(cg, 'nx_graph')
        if nxg is not None:
            name_to_idx = {}
            if var_cols is not None:
                name_to_idx = {str(n): i for i, n in enumerate(var_cols)}
            labels = getattr(cg, 'labels', None)
            if labels and (not name_to_idx):
                name_to_idx = {str(n): i for i, n in enumerate(labels)}
            if nxg.is_directed():
                for u, v in nxg.edges():
                    su, sv = (str(u), str(v))
                    if isinstance(u, int) and var_cols and (u < len(var_cols)):
                        iu = u
                    else:
                        iu = name_to_idx.get(su)
                    if isinstance(v, int) and var_cols and (v < len(var_cols)):
                        iv = v
                    else:
                        iv = name_to_idx.get(sv)
                    if iu is None or iv is None:
                        continue
                    edges_raw.append((iu, iv, '->'))
            else:
                for u, v in nxg.edges():
                    su, sv = (str(u), str(v))
                    iu = None
                    iv = None
                    if isinstance(u, int) and var_cols and (u < len(var_cols)):
                        iu = u
                    else:
                        iu = name_to_idx.get(su)
                    if isinstance(v, int) and var_cols and (v < len(var_cols)):
                        iv = v
                    else:
                        iv = name_to_idx.get(sv)
                    if iu is None or iv is None:
                        continue
                    edges_raw.append((iu, iv, '---'))
            return edges_raw
    except Exception:
        pass
    for attr in ['G', 'graph']:
        try:
            G = getattr(cg, attr)
            s = str(G)
            ms = re.findall('X(\\d+).*?(<->|<--|-->|o->|<-o|o-o|---).*?X(\\d+)', s)
            for a, pat, b in ms:
                i = max(0, int(a) - 1)
                j = max(0, int(b) - 1)
                edges_raw.append((i, j, pat))
            if edges_raw:
                return edges_raw
        except Exception:
            continue
    return edges_raw
    return edges_raw

def run_fci(np, df_train, var_cols: List[str], alpha: float, out_png: str, out_adj: str, debug_path: Optional[str]=None):
    fci, fisherz, _ = try_import_causallearn()
    if fci is None:
        print('未检测到 causal-learn，将无法运行 FCI。请安装：pip install causal-learn', file=sys.stderr)
        raise RuntimeError('causal-learn missing')
    data = df_train[var_cols].to_numpy(dtype=float)
    data, _, _ = zscore(None, np, data)
    res = fci(data, fisherz, alpha)
    cg = res[0] if isinstance(res, tuple) else res
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    edges_raw = _parse_edges_generic(cg, var_cols)
    edges_idx: List[Tuple[int, int, str]] = []
    for i, j, pat in edges_raw:
        if i < len(var_cols) and j < len(var_cols):
            edges_idx.append((i, j, pat))
    G = nx.DiGraph()
    for i, name in enumerate(var_cols):
        G.add_node(i, label=name)
    for i, j, pat in edges_idx:
        if pat in ('->', 'o->'):
            G.add_edge(i, j)
        elif pat in ('<-', '<-o'):
            G.add_edge(j, i)
        else:
            G.add_edge(i, j)
            G.add_edge(j, i)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(max(8, len(var_cols) * 0.4), max(6, len(var_cols) * 0.3)))
    nx.draw(G, pos, with_labels=True, labels={i: n for i, n in enumerate(var_cols)}, node_size=500, font_size=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    import pandas as pd
    T, H, C = build_endpoint_matrices(var_cols, edges_idx)
    A_dir = endpoints_to_directed_adj(np, T, H)
    A_skel = endpoints_to_skeleton(np, T, H, C)
    A_semi = endpoints_to_semi_directed(np, T, H, C)
    base = os.path.splitext(out_adj)[0]
    pd.DataFrame(T, index=var_cols, columns=var_cols).to_csv(base + '_tail.csv', encoding='utf-8-sig')
    pd.DataFrame(H, index=var_cols, columns=var_cols).to_csv(base + '_arrow.csv', encoding='utf-8-sig')
    pd.DataFrame(C, index=var_cols, columns=var_cols).to_csv(base + '_circle.csv', encoding='utf-8-sig')
    pd.DataFrame(A_dir, index=var_cols, columns=var_cols).to_csv(out_adj, encoding='utf-8-sig')
    pd.DataFrame(A_skel, index=var_cols, columns=var_cols).to_csv(base + '_skeleton.csv', encoding='utf-8-sig')
    pd.DataFrame(A_semi, index=var_cols, columns=var_cols).to_csv(base + '_semi.csv', encoding='utf-8-sig')
    if debug_path:
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f'FCI type: {type(cg)}\n')
                try:
                    f.write(f'CG attrs: {dir(cg)}\n')
                except Exception:
                    pass
                f.write(f'var_cols(n={len(var_cols)}): {var_cols}\n')
                f.write(f'edges_raw(n={len(edges_raw)}): {edges_raw}\n')
                f.write(f'edges_idx(n={len(edges_idx)}): {edges_idx}\n')
                try:
                    f.write(f'adj(sum)={int((A_dir != 0).sum())}\n')
                except Exception:
                    pass
        except Exception:
            pass
    return {'adj': A_dir, 'skeleton': A_skel, 'semi': A_semi, 'tail': T, 'arrow': H, 'circle': C, 'edges': edges_idx}

def run_pc(np, df_train, var_cols: List[str], alpha: float, out_png: str, out_adj: str, debug_path: Optional[str]=None):
    _, fisherz, pc = try_import_causallearn()
    if pc is None:
        print('未检测到 PC 算法入口（causal-learn），请安装或更新：pip install causal-learn', file=sys.stderr)
        raise RuntimeError('causal-learn PC missing')
    data = df_train[var_cols].to_numpy(dtype=float)
    data, _, _ = zscore(None, np, data)
    try:
        res = pc(data, indep_test='fisherz', alpha=alpha)
    except TypeError:
        res = pc(data, fisherz, alpha)
    cg = res[0] if isinstance(res, tuple) else res
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    edges_raw = _parse_edges_generic(cg, var_cols)
    N = len(var_cols)
    edges_idx: List[Tuple[int, int, str]] = []
    for i, j, pat in edges_raw:
        if i >= N or j >= N:
            continue
        edges_idx.append((i, j, pat))
    T, H, C = build_endpoint_matrices(var_cols, edges_idx)
    A_dir = endpoints_to_directed_adj(np, T, H)
    A_skel = endpoints_to_skeleton(np, T, H, C)
    A_semi = endpoints_to_semi_directed(np, T, H, C)
    G = nx.Graph()
    for k, name in enumerate(var_cols):
        G.add_node(k, label=name)
    for i in range(N):
        for j in range(i + 1, N):
            present = T[i, j] or T[j, i] or H[i, j] or H[j, i] or C[i, j] or C[j, i]
            if present:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(max(8, N * 0.4), max(6, N * 0.3)))
    nx.draw(G, pos, with_labels=True, labels={i: n for i, n in enumerate(var_cols)}, node_size=500, font_size=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    import pandas as pd
    base = os.path.splitext(out_adj)[0]
    pd.DataFrame(T, index=var_cols, columns=var_cols).to_csv(base + '_tail.csv', encoding='utf-8-sig')
    pd.DataFrame(H, index=var_cols, columns=var_cols).to_csv(base + '_arrow.csv', encoding='utf-8-sig')
    pd.DataFrame(C, index=var_cols, columns=var_cols).to_csv(base + '_circle.csv', encoding='utf-8-sig')
    pd.DataFrame(A_dir, index=var_cols, columns=var_cols).to_csv(out_adj, encoding='utf-8-sig')
    pd.DataFrame(A_skel, index=var_cols, columns=var_cols).to_csv(base + '_skeleton.csv', encoding='utf-8-sig')
    pd.DataFrame(A_semi, index=var_cols, columns=var_cols).to_csv(base + '_semi.csv', encoding='utf-8-sig')
    if debug_path:
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f'PC type: {type(cg)}\n')
                try:
                    f.write(f'CG attrs: {dir(cg)}\n')
                except Exception:
                    pass
                f.write(f'var_cols(n={len(var_cols)}): {var_cols}\n')
                f.write(f'edges_raw(n={len(edges_raw)}): {edges_raw}\n')
                try:
                    f.write(f'adj(sum)={int((A_dir != 0).sum())}\n')
                except Exception:
                    pass
        except Exception:
            pass
    return {'adj': A_dir, 'skeleton': A_skel, 'semi': A_semi, 'tail': T, 'arrow': H, 'circle': C, 'edges': edges_idx}

class DAGLinear(torch.nn.Module):
    """
    线性版本的 DAG 结构学习（更接近 NOTEARS），避免 MLP 引入的缩放不识别性：
    - 学习 A ∈ [0,1]^{N×N}（sigmoid 映射），对角线为 0。
    - 使用线性重构 X_hat = X @ A。
    - 损失 = ||X - X_hat||^2 + λ_acyc * h(A) + λ_sparse * |A|。
    """

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.S_param = torch.nn.Parameter(torch.randn(N, N) * 0.1)
        self.W_param = torch.nn.Parameter(torch.randn(N, N) * 0.1)
        self.register_buffer('mask_no_diag', 1 - torch.eye(N))
        self.tau = 1.0

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def adjacency(self):
        M = torch.sigmoid(self.S_param * self.tau)
        A_eff = M * self.W_param * self.mask_no_diag
        return A_eff

    def forward(self, X):
        A = self.adjacency()
        X_hat = torch.matmul(X, A)
        return (X_hat, A)

def _acyclicity_penalty_torch(torch, A):
    S = torch.abs(A)
    S = S * S
    E = torch.matrix_exp(S)
    h = torch.trace(E) - A.shape[0]
    return h

class VAED(torch.nn.Module):
    """简化版 VAED：用于学习潜空间并与GMM交替优化，输出 q(z|x) 的 μ/σ。"""

    def __init__(self, in_dim: int, hidden: int, z_dim: int):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.mu = torch.nn.Linear(hidden, z_dim)
        self.logvar = torch.nn.Linear(hidden, z_dim)
        self.dec = torch.nn.Sequential(torch.nn.Linear(z_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, in_dim))

    def encode(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return (mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return (x_hat, mu, logvar, z)

def _vaed_train_and_cluster(torch, np, X_np, K: int, z_dim: int, hidden: int, epochs: int, lr: float, lambda3: float, gmm_update_every: int, seed: int=42, station_labels: Any=None, outdir: Any=None):
    """
    训练VAED并交替拟合GMM：
    - X_np: [B, N]（已标准化特征更稳）
    - 返回：z_mu, z_logvar, resp(N×K职责), gmm(均值、精度) 的简化结构
    """
    device = torch.device('cpu')
    B, N = X_np.shape
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    model = VAED(in_dim=N, hidden=hidden, z_dim=z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss = torch.nn.MSELoss(reduction='mean')
    rng = np.random.default_rng(seed)
    for ep in range(max(0, int(getattr((args_if_exists := type('obj', (), {})()), 'vaed_pretrain', 0)))):
        pass

    def e_step(mu_np, logvar_np):
        Kloc = gmm['eta'].shape[0]
        D = gmm['eta'].shape[1]
        tau = gmm['tau'][None, :]
        pi = gmm['pi'][None, :]
        mu = mu_np[:, None, :]
        eta = gmm['eta'][None, :, :]
        sq = np.sum((mu - eta) ** 2, axis=2)
        logit = np.log(pi + 1e-08) - 0.5 * (tau * sq)
        logit = logit - logit.max(axis=1, keepdims=True)
        resp = np.exp(logit)
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-08)
        return resp.astype(np.float32)

    def m_step(mu_np, resp, alpha0=1.0, tau_max=100.0):
        Nk = resp.sum(axis=0) + 1e-06
        pi = (Nk + alpha0) / (Nk.sum() + K * alpha0)
        eta = resp.T @ mu_np / Nk[:, None]
        diff2 = (mu_np[:, None, :] - eta[None, :, :]) ** 2 * resp[:, :, None]
        mse = diff2.sum(axis=(0, 2)) / (Nk * mu_np.shape[1] + 1e-06)
        tau = 1.0 / (mse + 1e-06)
        tau = np.clip(tau, 1.0, tau_max)
        return {'eta': eta.astype(np.float32), 'tau': tau.astype(np.float32), 'pi': pi.astype(np.float32)}
    pre_ep = 0
    if 'pretrain_steps' not in locals():
        pre_ep = 20
    for _ in range(pre_ep):
        model.train()
        opt.zero_grad()
        x_hat, mu_t, logvar_t, _ = model(X_t)
        rec = recon_loss(x_hat, X_t)
        kld = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
        (rec + kld).backward()
        opt.step()
    with torch.no_grad():
        _, mu_t, logvar_t, _ = model(X_t)
    mu_np0 = mu_t.cpu().numpy()
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=seed)
        lbl = km.fit_predict(mu_np0)
        eta0 = km.cluster_centers_.astype(np.float32)
        Nk0 = np.bincount(lbl, minlength=K) + 1e-06
        pi0 = Nk0 / Nk0.sum()
        diff2 = (mu_np0 - eta0[lbl]) ** 2
        mse0 = []
        for k in range(K):
            sel = lbl == k
            if sel.sum() == 0:
                mse0.append(1.0)
            else:
                mse0.append(float(diff2[sel].mean()))
        tau0 = 1.0 / (np.array(mse0) + 1e-06)
        tau0 = np.clip(tau0, 1.0, 100.0)
        gmm = {'eta': eta0, 'tau': tau0.astype(np.float32), 'pi': pi0.astype(np.float32)}
    except Exception:
        Z_mu_init = rng.normal(size=(K, z_dim)).astype(np.float32)
        tau_init = np.ones((K,), dtype=np.float32)
        pi_init = np.ones((K,), dtype=np.float32) / K
        gmm = {'eta': Z_mu_init.copy(), 'tau': tau_init.copy(), 'pi': pi_init.copy()}
    best = {'loss': float('inf'), 'state': None}
    loss_hist: List[float] = []
    rec_hist: List[float] = []
    kld_hist: List[float] = []
    dist_hist: List[float] = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        x_hat, mu_t, logvar_t, z_t = model(X_t)
        rec = recon_loss(x_hat, X_t)
        kld = -0.5 * torch.mean(1 + logvar_t - mu_t.pow(2) - logvar_t.exp())
        with torch.no_grad():
            mu_np = mu_t.detach().cpu().numpy()
            logvar_np = logvar_t.detach().cpu().numpy()
            resp = e_step(mu_np, logvar_np)
            k_idx = np.argmax(resp, axis=1)
            eta_sel = gmm['eta'][k_idx]
            tau_sel = gmm['tau'][k_idx]
        sigma2_t = torch.clamp(torch.exp(logvar_t), min=0.001)
        eta_t = torch.tensor(eta_sel, dtype=torch.float32, device=device)
        tau_t = torch.tensor(tau_sel, dtype=torch.float32, device=device)
        dist = 0.5 * (torch.log(tau_t + 1e-08).unsqueeze(-1) - torch.log(sigma2_t + 1e-08) + (sigma2_t + (mu_t - eta_t) ** 2) * tau_t.unsqueeze(-1) - 1.0)
        dist = dist.mean()
        warm = min(1.0, (ep + 1) / max(1.0, epochs * 0.3))
        loss = rec + kld + lambda3 * warm * dist
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        try:
            loss_hist.append(float(loss.detach().cpu()))
            rec_hist.append(float(rec.detach().cpu()))
            kld_hist.append(float(kld.detach().cpu()))
            dist_hist.append(float(dist.detach().cpu()))
        except Exception:
            pass
        if (ep + 1) % max(1, gmm_update_every) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                _, mu_t, logvar_t, _ = model(X_t)
            mu_np = mu_t.cpu().numpy()
            resp = e_step(mu_np, logvar_t.cpu().numpy())
            gmm = m_step(mu_np, resp, alpha0=1.0, tau_max=100.0)
            assign = resp.argmax(axis=1)
            counts = [(k, int((assign == k).sum())) for k in range(K)]
            counts_str = ', '.join([f'k{k}={v}' for k, v in counts])
            print(f'[VAED] epoch {ep + 1}: cluster sizes: {counts_str}')
            if station_labels is not None:
                try:
                    for k in range(K):
                        idx = np.where(assign == k)[0]
                        if idx.size == 0:
                            continue
                        labs, cnts = np.unique(station_labels[idx], return_counts=True)
                        order = np.argsort(cnts)[::-1][:2]
                        top = ', '.join([f'{labs[o]}:{int(cnts[o])}' for o in order])
                        print(f'  - k{k}: top sites => {top}')
                except Exception:
                    pass
        if float(loss.item()) < best['loss']:
            best = {'loss': float(loss.item()), 'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 'gmm': {k: v.copy() for k, v in gmm.items()}}
    model.load_state_dict(best['state'])
    gmm = best['gmm']
    model.eval()
    with torch.no_grad():
        _, mu_t, logvar_t, _ = model(X_t)
    mu_np = mu_t.cpu().numpy()
    logvar_np = logvar_t.cpu().numpy()
    resp = e_step(mu_np, logvar_np)
    if station_labels is not None and outdir is not None:
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            Z2 = pca.fit_transform(mu_np)
            sites = np.unique(station_labels)
            site_to_idx = {s: i for i, s in enumerate(sites)}
            colors = np.array([site_to_idx[s] for s in station_labels])
            cmap = plt.cm.get_cmap('tab20', len(sites))
            plt.figure(figsize=(8.5, 6))
            sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=colors, cmap=cmap, s=12, alpha=0.85, edgecolors='none')
            plt.title('VAED latent (colored by station)')
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(sites[i]), markerfacecolor=cmap(i), markeredgecolor='none', markersize=6) for i in range(len(sites))]
            ncol = 1 if len(sites) <= 10 else 2 if len(sites) <= 20 else 3
            leg = plt.legend(handles=handles, title='生态站', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0, fontsize=8, frameon=False, ncol=ncol)
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            plt.savefig(os.path.join(outdir, 'vaed_latent_by_station.png'), dpi=200)
            plt.close()
        except Exception:
            pass
    try:
        if outdir is not None:
            import pandas as pd
            assign_final = resp.argmax(axis=1)
            pd.DataFrame({'cluster': assign_final}).to_csv(os.path.join(outdir, 'vaed_assignments.csv'), index=False, encoding='utf-8-sig')
            try:
                df_loss = pd.DataFrame({'epoch': list(range(1, len(loss_hist) + 1)), 'loss': loss_hist, 'recon': rec_hist, 'kld': kld_hist, 'dist': dist_hist})
                df_loss.to_csv(os.path.join(outdir, 'vaed_train_loss.csv'), index=False, encoding='utf-8-sig')
                import matplotlib.pyplot as _plt
                _plt.figure(figsize=(7, 4))
                _plt.plot(df_loss['epoch'], df_loss['loss'], label='total')
                _plt.plot(df_loss['epoch'], df_loss['recon'], label='recon', alpha=0.7)
                _plt.plot(df_loss['epoch'], df_loss['kld'], label='kld', alpha=0.7)
                _plt.plot(df_loss['epoch'], df_loss['dist'], label='dist', alpha=0.7)
                _plt.xlabel('epoch')
                _plt.ylabel('loss')
                _plt.title('VAED training loss')
                _plt.legend(frameon=False, fontsize=8)
                _plt.tight_layout()
                _plt.savefig(os.path.join(outdir, 'vaed_train_loss.png'), dpi=200)
                _plt.close()
                try:
                    import shutil as _sh
                    _sh.copyfile(os.path.join(outdir, 'vaed_train_loss.png'), os.path.join(outdir, 'avaed_train_loss.png'))
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    return {'mu': mu_np, 'logvar': logvar_np, 'resp': resp, 'gmm': gmm, 'model': model}

def _load_ivae_class():
    """动态加载 models/ivae_dag_gnn.py 中的 iVAE_DAG_GNN 类。"""
    import importlib.util, os as _os
    model_path = _os.path.join(_os.path.dirname(__file__), 'models', 'ivae_dag_gnn.py')
    if not _os.path.exists(model_path):
        raise RuntimeError(f'未找到 {model_path}，无法运行 iVAE 实验组')
    spec = importlib.util.spec_from_file_location('ivae_dag_gnn', model_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'iVAE_DAG_GNN'):
        raise RuntimeError('iVAE 模块缺少 iVAE_DAG_GNN 类')
    return mod.iVAE_DAG_GNN

def _encode_u_with_unk(pd, series, train_index):
    """仅用训练索引拟合类别到ID映射；验证/测试未见类别映射到 UNK。返回 (ids, encoder_dict, n_levels)。"""
    ser = series.astype(str).fillna('')
    train_vals = ser.loc[train_index].unique().tolist()
    enc = {v: i for i, v in enumerate(sorted(train_vals))}
    unk_id = len(enc)
    ids = ser.apply(lambda v: enc.get(v, unk_id)).astype(int)
    n_levels = unk_id + 1
    return (ids, enc, n_levels)

def _plot_latent_causal_graph(np, nx, plt, A, out_png: str, labels_override: Optional[Dict[int, str]]=None):
    """基于 |A|>thr 的规则绘制 Z 因果图（threshold 在外部已经应用）。
    labels_override: 可选，以 {node_index: "Zk(var1,var2,...)"} 覆盖默认标签。
    """
    N = A.shape[0]
    G = nx.DiGraph()
    labels = {i: labels_override.get(i) if labels_override and i in labels_override else f'Z{i + 1}' for i in range(N)}
    for i in range(N):
        G.add_node(i, label=labels[i])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if A[i, j] != 0:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(max(8, N * 0.4), max(6, N * 0.3)))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, font_size=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def _save_ivae_adjacency(np, pd, A_cont, out_adj_csv: str, thr_abs: float):
    base = os.path.splitext(out_adj_csv)[0]
    pd.DataFrame(A_cont).to_csv(base + '_continuous.csv', index=False, header=False, encoding='utf-8-sig')
    A_bin = (np.abs(A_cont) >= float(thr_abs)).astype(int)
    np.fill_diagonal(A_bin, 0)
    pd.DataFrame(A_bin).to_csv(out_adj_csv, index=False, header=False, encoding='utf-8-sig')
    return A_bin

def _compute_z_alias_labels(np, Xtr, Ztr, var_cols: List[str], top_k: int=3):
    X = np.asarray(Xtr, dtype=float)
    Z = np.asarray(Ztr, dtype=float)
    n = X.shape[0]
    Xm = X - np.nanmean(X, axis=0, keepdims=True)
    Xs = np.nanstd(Xm, axis=0, keepdims=True) + 1e-08
    Xn = Xm / Xs
    Zm = Z - np.nanmean(Z, axis=0, keepdims=True)
    Zs = np.nanstd(Zm, axis=0, keepdims=True) + 1e-08
    Zn = Zm / Zs
    C = Zn.T @ Xn / float(max(1, n - 1))
    Zdim = C.shape[0]
    labels = {}
    details = []
    for k in range(Zdim):
        row = C[k]
        idx = np.argsort(-np.abs(row))[:max(1, int(top_k))]
        names = [var_cols[j] for j in idx]
        labels[k] = f'Z{k + 1}(' + ', '.join(names) + ')'
        for r, j in enumerate(idx, start=1):
            details.append({'Z': f'Z{k + 1}', 'rank': r, 'var': var_cols[j], 'corr': float(row[j])})
    return (labels, details)

def _save_z_alias_csv(pd, details_rows: List[Dict], out_csv: str):
    df = pd.DataFrame(details_rows)
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')

def _gmm_responsibilities_from_mu(np, mu_np, gmm):
    # 给定 mu_np[B,D] 与 GMM 参数，计算职责分布 resp[B,K]。
    K = gmm['eta'].shape[0]
    tau = gmm['tau'][None, :]
    pi = gmm['pi'][None, :]
    mu = mu_np[:, None, :]
    eta = gmm['eta'][None, :, :]
    sq = np.sum((mu - eta) ** 2, axis=2)
    logit = np.log(pi + 1e-08) - 0.5 * (tau * sq)
    logit = logit - logit.max(axis=1, keepdims=True)
    resp = np.exp(logit)
    resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-08)
    return resp.astype(np.float32)

def _plot_variable_causal_graph(np, nx, plt, A_bin, var_cols: List[str], out_png: str):
    # 绘制基于变量名的因果图（A_bin 为二值邻接，0/1）。
    N = A_bin.shape[0]
    G = nx.DiGraph()
    labels = {i: var_cols[i] if i < len(var_cols) else f'X{i + 1}' for i in range(N)}
    for i in range(N):
        G.add_node(i, label=labels[i])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if A_bin[i, j] != 0:
                G.add_edge(i, j)
    max_nodes_for_plot = 40
    if N > max_nodes_for_plot:
        nodes_to_keep = list(range(max_nodes_for_plot))
        G = G.subgraph(nodes_to_keep).copy()
        labels = {i: labels[i] for i in nodes_to_keep}
    try:
        pos = nx.shell_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#F5F5F5', edgecolors='#333333', linewidths=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=10, width=0.6, edge_color='#555555', alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

def _edge_count_from_adj(np, adj):
    A = (np.abs(adj) > 0).astype(int)
    N = A.shape[0]
    if np.allclose(A, A.T):
        return int(np.triu(A, 1).sum())
    else:
        return int(A.sum() - np.trace(A))

def _row_normalize(np, A):
    A = np.asarray(A, dtype=float)
    rs = A.sum(axis=1, keepdims=True) + 1e-08
    return A / rs

def _graph_poly_features(np, X, A_core, orders: List[int], mode: str='both'):
    """
    计算多项式图滤波特征并做池化（mean/std）。
    - X: [B, N]
    - A_core: [N, N]（未加自环，0/1或权重均可）
    - orders: 阶数列表（正整数）
    - mode: 'A'（沿A传播）, 'AT'（沿A^T传播）, 'sym'（对称化后传播）, 'both'（A和AT都做并拼接）
    返回：GF [B, D]，其中 D = 2 * (len(orders) * n_modes)
    """
    N = X.shape[1]
    A = np.asarray(A_core, dtype=float)
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    mats = []
    if mode in ('A', 'both'):
        mats.append(_row_normalize(np, A))
    if mode in ('AT', 'both'):
        mats.append(_row_normalize(np, A.T))
    if mode == 'sym':
        As = (A + A.T > 0).astype(float)
        mats.append(_row_normalize(np, As))
    feats = []
    for M in mats:
        Mk = np.eye(N, dtype=float)
        for k in range(1, max(orders) + 1):
            Mk = Mk @ M
            if k in orders:
                Hk = X @ Mk
                mean_k = Hk.mean(axis=1, keepdims=True)
                std_k = Hk.std(axis=1, keepdims=True)
                feats.append(mean_k)
                feats.append(std_k)
    if feats:
        GF = np.concatenate(feats, axis=1)
    else:
        GF = np.zeros((X.shape[0], 0), dtype=float)
    return GF

def _aggregate_cluster_dags(np, var_cols: List[str], work_tr_df, resp, args, dag_runner, agg_threshold: Optional[float]=None, prefix: str='vaed'):
    # 按簇权重在子集上学习 DAG，再按簇权重聚合：
    # resp: [B, K] 每行是样本对簇的责任概率（或注意力权重）
    K = resp.shape[1]
    N = len(var_cols)
    A_accum_prop = np.zeros((N, N), dtype=float)
    A_accum_cont = np.zeros((N, N), dtype=float)
    assign = resp.argmax(axis=1)
    weights = resp.mean(axis=0)
    weights = weights / (weights.sum() + 1e-08)
    for k in range(K):
        idx = np.where(assign == k)[0]
        if idx.size < 5:
            continue
        df_k = work_tr_df.iloc[idx]
        try:
            out_png = os.path.join(args.outdir, f'{prefix}_dag_gnn_graph_k{k}.png')
            out_adj = os.path.join(args.outdir, f'{prefix}_dag_gnn_adjacency_k{k}.csv')
            edges_dbg = os.path.join(args.outdir, f'{prefix}_dag_gnn_edges_k{k}.txt')
            A_k_bin = run_dag_gnn(torch, np, df_k, var_cols, hidden=args.dag_hidden, epochs=max(100, args.dag_epochs // 2), lr=args.dag_lr, lambda_acyc=args.dag_lambda_acyc, lambda_sparse=args.dag_lambda_sparse, threshold=args.dag_threshold, out_png=out_png, out_adj=out_adj, edges_dbg_out=edges_dbg, cat_cols=list(work_tr_df.columns.intersection(dag_runner['cat_cols'])) if 'cat_cols' in dag_runner else None)
            try:
                import pandas as _pd
                base, ext = os.path.splitext(out_adj)
                cont_path = base + '_continuous.csv'
                if os.path.exists(cont_path):
                    A_k_cont = _pd.read_csv(cont_path, header=None).to_numpy(dtype=float)
                else:
                    A_k_cont = A_k_bin.astype(float)
            except Exception:
                A_k_cont = A_k_bin.astype(float)
        except Exception:
            continue
        A_accum_prop += float(weights[k]) * A_k_bin.astype(float)
        A_accum_cont += float(weights[k]) * A_k_cont.astype(float)
    thr = float(agg_threshold if agg_threshold is not None else args.vaed_agg_threshold)
    A_bin = (A_accum_prop >= thr).astype(int)
    np.fill_diagonal(A_bin, 0)
    return (A_bin, A_accum_cont, A_accum_prop)

def _binarize_adj_with_budget(np, A_cont, base_threshold: Optional[float]=None, target_edges: int=0):
    # 基于绝对值阈值与目标边数对连续邻接矩阵进行二值化
    # 若 target_edges > 0：在去对角线的 |A| 上选择分位数阈值，使得边数≈target_edges；
    A = np.asarray(A_cont).copy()
    n = A.shape[0]
    mask_off = ~np.eye(n, dtype=bool)
    vals = np.abs(A)[mask_off]
    thr_eff = float(base_threshold) if base_threshold is not None else 0.0
    if target_edges is not None and int(target_edges) > 0 and (vals.size > 0):
        k = int(target_edges)
        k = max(1, min(k, vals.size))
        t_k = float(np.partition(vals, vals.size - k)[vals.size - k])
        thr_eff = max(thr_eff, t_k)
    A_bin = (np.abs(A) >= thr_eff).astype(int)
    np.fill_diagonal(A_bin, 0)
    return (A_bin, thr_eff, int(A_bin.sum()))

def _compute_attention_responsibilities(np, mu_np, gmm, gamma: float=0.5, beta: float=1.0, temp_fine: float=1.0, temp_coarse: float=2.0):
    # 基于 VAED 的潜表示 mu 和 GMM 参数，构造聚类注意力与层次注意力的融合注意力
    eta = gmm['eta'].astype(np.float32)
    tau = gmm['tau'].astype(np.float32)
    K, D = (eta.shape[0], mu_np.shape[1])
    a = tau / (tau.sum() + 1e-08)
    loga = np.log(a + 1e-08)
    scores_fine = mu_np @ eta.T / max(1e-06, temp_fine)
    scores_fine = scores_fine + loga[None, :]
    scores_fine = scores_fine - scores_fine.max(axis=1, keepdims=True)
    attn_fine = np.exp(scores_fine)
    attn_fine = attn_fine / (attn_fine.sum(axis=1, keepdims=True) + 1e-08)
    try:
        from sklearn.decomposition import PCA
        p = max(2, D // 2)
        pca = PCA(n_components=p, random_state=42)
        Z_mu = pca.fit_transform(mu_np)
        Z_eta = pca.transform(eta)
        scores_coarse = Z_mu @ Z_eta.T / max(1e-06, temp_coarse)
        scores_coarse = scores_coarse + loga[None, :]
        scores_coarse = scores_coarse - scores_coarse.max(axis=1, keepdims=True)
        attn_coarse = np.exp(scores_coarse)
        attn_coarse = attn_coarse / (attn_coarse.sum(axis=1, keepdims=True) + 1e-08)
    except Exception:
        attn_coarse = attn_fine.copy()
    attn = gamma * attn_fine + (1.0 - gamma) * attn_coarse
    attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-08)
    return attn

def _plot_cluster_station_stack(pd, np, resp, station_labels, outdir: str, fname: str='vaed_cluster_station_stack.png'):
    try:
        import matplotlib.pyplot as plt
        assign = resp.argmax(axis=1)
        K = resp.shape[1]
        clusters = [f'k{i}' for i in range(K)]
        stations = np.unique(station_labels)
        data = np.zeros((K, len(stations)), dtype=float)
        for k in range(K):
            idx = np.where(assign == k)[0]
            if idx.size == 0:
                continue
            labs, cnts = np.unique(station_labels[idx], return_counts=True)
            for s, c in zip(labs, cnts):
                j = np.where(stations == s)[0][0]
                data[k, j] = c
            row_sum = data[k].sum()
            if row_sum > 0:
                data[k] = data[k] / row_sum
        x = np.arange(K)
        bottom = np.zeros(K, dtype=float)
        cmap = plt.cm.get_cmap('tab20', len(stations))
        plt.figure(figsize=(max(8, K * 1.0), 6))
        bars = []
        for j, s in enumerate(stations):
            bar = plt.bar(x, data[:, j], bottom=bottom, color=cmap(j), edgecolor='none', width=0.8, label=str(s))
            bars.append(bar)
            bottom = bottom + data[:, j]
        plt.xticks(x, clusters)
        plt.ylabel('站点占比')
        plt.title('各簇的生态站占比（堆叠）')
        ncol = 1 if len(stations) <= 10 else 2 if len(stations) <= 20 else 3
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8, ncol=ncol, title='生态站')
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(os.path.join(outdir, fname), dpi=200)
        plt.close()
    except Exception:
        pass

def _plot_cluster_feature_stats(pd, np, resp, df_train, var_cols, target_col: str, outdir: str, top_n: int=8, fname_radar: str='vaed_cluster_feature_radar.png', fname_bars: str='vaed_cluster_feature_bars.png'):
    try:
        import matplotlib.pyplot as plt
        cols = [c for c in var_cols if c != target_col]
        X = df_train[cols].to_numpy(dtype=float)
        X = np.where(np.isnan(X), 0.0, X)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std < 1e-08] = 1.0
        Xz = (X - mean) / std
        assign = resp.argmax(axis=1)
        K = resp.shape[1]
        Cmean = np.zeros((K, len(cols)), dtype=float)
        Cstd = np.zeros((K, len(cols)), dtype=float)
        counts = np.zeros((K,), dtype=int)
        for k in range(K):
            idx = np.where(assign == k)[0]
            counts[k] = idx.size
            if idx.size > 0:
                Cmean[k] = np.nanmean(Xz[idx], axis=0)
                Cstd[k] = np.nanstd(Xz[idx], axis=0)
        var_across_clusters = np.var(Cmean, axis=0)
        top_idx = np.argsort(var_across_clusters)[::-1][:min(top_n, len(cols))]
        feats = [cols[i] for i in top_idx]
        topK = int(min(K, 5))
        order = np.argsort(counts)[::-1][:topK]
        angles = np.linspace(0, 2 * np.pi, len(top_idx), endpoint=False).tolist()
        angles += angles[:1]
        plt.figure(figsize=(7.5, 6.5))
        ax = plt.subplot(111, polar=True)
        cmap = plt.cm.get_cmap('tab10', topK)
        for r, k in enumerate(order):
            vals = Cmean[k, top_idx].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, color=cmap(r), linewidth=2, label=f'k{k} (n={counts[k]})')
            ax.fill(angles, vals, color=cmap(r), alpha=0.15)
        ax.set_xticks(np.linspace(0, 2 * np.pi, len(top_idx), endpoint=False))
        ax.set_xticklabels(feats, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title('不同簇的关键特征均值（标准化后）- 雷达图')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(outdir, fname_radar), dpi=200)
        plt.close()
        plt.figure(figsize=(max(8, len(top_idx) * 1.0), 6))
        width = 0.8 / topK
        x = np.arange(len(top_idx))
        for r, k in enumerate(order):
            means = Cmean[k, top_idx]
            errs = Cstd[k, top_idx]
            plt.bar(x + r * width, means, yerr=errs, width=width, label=f'k{k} (n={counts[k]})', alpha=0.8)
        plt.xticks(x + (topK - 1) * width / 2, feats, rotation=20)
        plt.ylabel('标准化均值 ± 标准差')
        plt.title('不同簇的关键特征统计 - 条形图')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.savefig(os.path.join(outdir, fname_bars), dpi=200)
        plt.close()
    except Exception:
        pass

def run_dag_gnn(torch, np, df_train, var_cols: List[str], hidden: int, epochs: int, lr: float, lambda_acyc: float, lambda_sparse: float, threshold: float, out_png: str, out_adj: str, edges_dbg_out: str, cat_cols: Optional[List[str]]=None):
    X_raw = df_train[var_cols].to_numpy(dtype=float)
    if cat_cols:
        cat_set = set(cat_cols)
        mask_np = np.zeros_like(X_raw, dtype=bool)
        for j, c in enumerate(var_cols):
            col_vals = X_raw[:, j]
            if c in cat_set:
                mask_np[:, j] = ~np.isnan(col_vals)
            else:
                mask_np[:, j] = ~np.isnan(col_vals) & (col_vals != 0)
    else:
        mask_np = ~np.isnan(X_raw) & (X_raw != 0)
    X_filled = np.where(mask_np, X_raw, 0.0)
    Xm = np.zeros(X_filled.shape[1], dtype=float)
    Xs = np.ones(X_filled.shape[1], dtype=float)
    for j in range(X_filled.shape[1]):
        m = mask_np[:, j]
        if m.any():
            Xm[j] = float(np.mean(X_filled[m, j]))
            Xs[j] = float(np.std(X_filled[m, j]) + 1e-08)
        else:
            Xm[j] = 0.0
            Xs[j] = 1.0
    Xn = (X_filled - Xm) / Xs
    device = torch.device('cpu')
    X_t = torch.tensor(Xn, dtype=torch.float32, device=device)
    M_t = torch.tensor(mask_np.astype(float), dtype=torch.float32, device=device)
    N = len(var_cols)
    model = DAGLinear(N).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    alpha = 0.0
    rho = 0.01
    update_ival = max(10, epochs // 8)
    best = {'obj': float('inf'), 'state': None}
    prev_h = None
    for ep in range(epochs):
        model.train()
        model.set_tau(1.0)
        opt.zero_grad()
        if ep % 2 == 1:
            X_in = X_t + 0.01 * torch.randn_like(X_t)
        else:
            X_in = X_t
        X_hat, A_eff = model(X_in)
        diff2 = (X_hat - X_in) ** 2
        denom = torch.clamp(M_t.sum(dim=0), min=1.0)
        rec_feat = (diff2 * M_t).sum(dim=0) / denom
        rec = 0.5 * rec_feat.mean()
        h = _acyclicity_penalty_torch(torch, A_eff)
        l1_eff = torch.sum(torch.abs(A_eff))
        l2_eff = torch.sum(A_eff * A_eff)
        obj = rec + lambda_sparse * l1_eff + 0.0001 * l2_eff + alpha * h + 0.5 * rho * (h * h)
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        if obj.item() < best['obj']:
            best['obj'] = float(obj.item())
            best['state'] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % update_ival == 0 or ep == epochs - 1:
            with torch.no_grad():
                _, A_chk = model(X_t)
                h_val = float(_acyclicity_penalty_torch(torch, A_chk).item())
            alpha += rho * h_val
            if prev_h is None or h_val > 0.8 * (prev_h if prev_h is not None else h_val + 1e-06):
                rho = min(rho * 2.0, 1000.0)
            prev_h = h_val
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        _, A_eff = model(X_t)
    A_np = A_eff.cpu().numpy()
    A_abs = np.abs(A_np)
    A_bin = (A_abs >= threshold).astype(int)
    np.fill_diagonal(A_bin, 0)
    if A_bin.sum() == 0:
        off = A_abs[~np.eye(N, dtype=bool)]
        q = float(np.quantile(off, 0.9)) if off.size > 0 else 0.0
        thr = max(threshold, q)
        if thr > 0:
            A_bin = (A_abs >= thr).astype(int)
            np.fill_diagonal(A_bin, 0)
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cont_path = os.path.join(os.path.dirname(out_adj), os.path.splitext(os.path.basename(out_adj))[0] + '_continuous.csv')
    pd.DataFrame(A_np, index=var_cols, columns=var_cols).to_csv(cont_path, encoding='utf-8-sig')
    pd.DataFrame(A_bin, index=var_cols, columns=var_cols).to_csv(out_adj, encoding='utf-8-sig')
    dbg = []
    for i in range(N):
        for j in range(N):
            if A_bin[i, j] != 0:
                dbg.append(f'{var_cols[i]} -> {var_cols[j]} : {A_np[i, j]:.4f}')
    try:
        with open(edges_dbg_out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dbg))
    except Exception:
        pass
    G = nx.DiGraph()
    for i, name in enumerate(var_cols):
        G.add_node(i, label=name)
    for i in range(N):
        for j in range(N):
            if A_bin[i, j] != 0:
                G.add_edge(i, j)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(max(8, N * 0.4), max(6, N * 0.3)))
    nx.draw(G, pos, with_labels=True, labels={i: n for i, n in enumerate(var_cols)}, node_size=500, font_size=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return A_bin