import os
# 为 CuBLAS 确定性设置工作区（需在任何 CUDA 调用之前设置）
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import json

from models import SharedSpecificNet
from utils import normalize_sparse_matrix, regularization_loss, dicr_loss, ZINB


class SimpleConfig:
    """使用"微调对比学习+"配置 - 基于超参数优化结果"""
    def __init__(self):
        # Model_Setup - 调整为与quick测试一致
        self.epochs = 150  # 与quick测试的最佳epoch范围一致
        self.lr = 0.001     # 保持原始学习率
        self.weight_decay = 5e-4  # 保持原始正则化
        # 允许通过环境变量覆盖（不改主逻辑，默认14）
        self.k = int(os.getenv('K_NEIGHBORS', '14'))
        self.radius = 550
        self.nhid1 = 128
        self.nhid2 = 64
        self.dropout = 0.1   # 保持原始dropout
        
        # 使用"微调对比学习+"配置 - 基于优化结果
        self.alpha = 0.88    # 表达重建损失（降低）
        self.beta = 0.62     # 结构重建损失（增加）
        self.gamma = 1.5     # 聚类损失（恢复最初基线）
        self.delta = 0.08    # 正则化损失（降低）
        self.dicr_weight = 0.055  # DICR损失权重（增加对比学习）
        self.theta = 0.3     # 特征图重建损失权重（保持）
        self.dom_weight = 0.0001  # 域判别损失权重（极小权重，最小干扰）
        self.orth_weight = 0.00001  # 正交约束损失权重（极小权重，最小干扰）
        
        self.no_cuda = False
        self.no_seed = False
        self.seed = 100
        
        # Data_Setting
        self.fdim = 3000
        
        # 新增参数
        self.proj_dim = 64
        
        # 聚类更新参数 - 恢复原始配置
        self.update_interval = 10  # 每10个epoch更新目标分布
        self.tol = 0.001  # 聚类变化容忍度
        
        # 图更新参数 - 恢复原始配置
        self.graph_update_interval = 20  # 每20个epoch使用ZINB mean更新特征图
        
        # 可视化参数
        self.save_plots = True  # 是否保存图像
        self.plot_interval = 50  # 每50个epoch保存一次中间结果
        self.figsize = (15, 5)  # 图像大小
        self.dpi = 300  # 图像分辨率
        self.lambda_distill = 0.0  # 知识蒸馏损失权重


def construct_spatial_graph(spatial_coords, radius=550):
    """构建空间图 G1: 基于欧氏距离"""
    print("Constructing spatial graph...")
    n_spots = spatial_coords.shape[0]
    
    # 计算欧氏距离
    distances = np.sqrt(np.sum((spatial_coords[:, np.newaxis] - spatial_coords[np.newaxis, :]) ** 2, axis=2))
    
    # 根据半径构建邻接矩阵
    A_spatial = (distances <= radius).astype(float)
    
    # 去除自环
    np.fill_diagonal(A_spatial, 0)
    
    print(f'Spatial graph: {np.sum(A_spatial)} edges, {n_spots} spots')
    print(f'Average neighbors per spot: {np.sum(A_spatial) / n_spots:.2f}')
    
    return A_spatial


def construct_feature_graph(features, k=14):
    """构建特征图 G2: 基于余弦相似度 + KNN（纯Torch实现）"""
    print("Constructing feature graph...")
    
    with torch.no_grad():
        x = features if torch.is_tensor(features) else torch.tensor(features)
        x = x.cpu()
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        sim = torch.mm(x, x.t())
        n = sim.size(0)
        sim.fill_diagonal_(-1.0)
        _, topk = torch.topk(sim, k=k, dim=1)
        rows = torch.arange(n).unsqueeze(1).expand_as(topk)
        A = torch.zeros((n, n), dtype=torch.float32)
        A[rows, topk] = 1.0
        A = torch.clamp(A + A.t(), max=1.0)
        A_feature = np.array(A.tolist(), dtype=float)
    
    print(f'Feature graph: {np.sum(A_feature)} edges, {features.shape[0]} spots')
    print(f'Average neighbors per spot: {np.sum(A_feature) / features.shape[0]:.2f}')
    
    return A_feature


def construct_feature_graph_from_mean(mean_reconstructed, k=14, cuda=False):
    # 纯Torch实现，避免numpy依赖
    if torch.is_tensor(mean_reconstructed):
        x = mean_reconstructed.detach().cpu()
    else:
        x = torch.tensor(mean_reconstructed, dtype=torch.float32)
    x = torch.nn.LayerNorm(x.shape[1])(x)
    
    with torch.no_grad():
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        sim = torch.mm(x, x.t())
        n = sim.size(0)
        sim.fill_diagonal_(-1.0)
        _, topk = torch.topk(sim, k=k, dim=1)
        rows = torch.arange(n).unsqueeze(1).expand_as(topk)
        A = torch.zeros((n, n), dtype=torch.float32)
        A[rows, topk] = 1.0
        A = torch.clamp(A + A.t(), max=1.0)
        A_feature = np.array(A.tolist(), dtype=float)
    
    print(f'ZINB-mean feature graph: {np.sum(A_feature)} edges, {n} spots')
    print(f'Average neighbors per spot: {np.sum(A_feature) / n:.2f}')
    
    # 归一化邻接矩阵
    A_feature_norm = normalize_sparse_matrix(sp.csr_matrix(A_feature) + sp.eye(A_feature.shape[0]))
    A_feature_norm = torch.FloatTensor(A_feature_norm.toarray())
    if cuda:
        A_feature_norm = A_feature_norm.cuda()
    return A_feature_norm, A_feature


def load_data(dataset):
    """加载数据并构建双视图图结构"""
    print(f"Loading data for dataset: {dataset}")
    
    # 使用原有的数据加载方式
    path = "./generate_data/DLPFC/" + dataset + "/MAFN.h5ad"
    adata = sc.read_h5ad(path)

    # 允许通过环境变量覆盖 PCA 维度（默认256），并控制是否使用 PCA 特征（默认启用）
    pca_dim = int(os.getenv('PCA_DIM', '256'))
    use_pca = os.getenv('USE_PCA_FEATURES', '1') in ('1', 'true', 'True')
    if use_pca:
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(adata.X)
        features = torch.FloatTensor(X_pca)
    else:
        features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    
    # 获取空间坐标（如果存在）
    if 'spatial' in adata.obsm:
        spatial_coords = adata.obsm['spatial']
    else:
        # 如果没有spatial坐标，尝试从其他地方获取
        print("Warning: No spatial coordinates found, using random coordinates")
        spatial_coords = np.random.rand(adata.n_obs, 2) * 1000
    
    print(f"Data shape: {features.shape}")
    print(f"Spatial coords shape: {spatial_coords.shape}")
    print(f"Labels shape: {len(labels)}")
    
    # 构建两个图的邻接矩阵（k 支持通过环境变量覆盖）
    A_spatial = construct_spatial_graph(spatial_coords, radius=550)
    k_neighbors = int(os.getenv('K_NEIGHBORS', '14'))
    A_feature = construct_feature_graph(features, k=k_neighbors)
    
    # 归一化邻接矩阵
    A_spatial_norm = normalize_sparse_matrix(sp.csr_matrix(A_spatial) + sp.eye(A_spatial.shape[0]))
    A_feature_norm = normalize_sparse_matrix(sp.csr_matrix(A_feature) + sp.eye(A_feature.shape[0]))
    
    A_spatial_norm = torch.FloatTensor(A_spatial_norm.toarray())
    A_feature_norm = torch.FloatTensor(A_feature_norm.toarray())
    
    # 构建正负样本对
    graph_nei = torch.FloatTensor(A_spatial + A_feature)
    graph_nei = (graph_nei > 0).float()
    graph_neg = torch.ones_like(graph_nei) - graph_nei
    
    print("Data loading completed!")
    return features, labels, A_spatial_norm, A_feature_norm, graph_nei, graph_neg


def zinb_loss_function(x_true, pi, disp, mean):
    """ZINB损失函数"""
    zinb = ZINB(pi, theta=disp, ridge_lambda=0)
    return zinb.loss(x_true, mean, mean=True)


def structure_loss_function(A_true, A_hat):
    """结构重建损失函数"""
    return F.mse_loss(A_hat, A_true)


def target_distribution(q, sharpen=2.0):  # 恢复最初基线
    q = q + 1e-8
    weight = q ** sharpen / (q.sum(0) + 1e-8)
    p = (weight.t() / (weight.sum(1) + 1e-8)).t()
    return p


# ================= 新增：域判别与正交损失 =================
class DomainDiscriminator(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super(DomainDiscriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2)
        )

    def forward(self, x):
        return self.net(x)


def domain_ce_loss(discriminator, s_s, s_f):
    """域判别交叉熵损失：确保特有表示真正分离"""
    logits_s = discriminator(s_s)
    logits_f = discriminator(s_f)
    labels_s = torch.zeros(s_s.shape[0], dtype=torch.long, device=s_s.device)
    labels_f = torch.ones(s_f.shape[0], dtype=torch.long, device=s_f.device)
    ce = torch.nn.functional.cross_entropy
    return (ce(logits_s, labels_s) + ce(logits_f, labels_f)) / 2.0


def orthogonality_loss(r_s, s_s, r_f, s_f):
    """正交约束损失：共享与特有表示去相关（归一化版本）"""
    # 归一化表示
    r_s_norm = F.normalize(r_s, p=2, dim=1)
    s_s_norm = F.normalize(s_s, p=2, dim=1)
    r_f_norm = F.normalize(r_f, p=2, dim=1)
    s_f_norm = F.normalize(s_f, p=2, dim=1)
    
    # 计算相关性（归一化后范围在[-1,1]）
    rs_ss_corr = torch.mm(r_s_norm.t(), s_s_norm)
    rf_sf_corr = torch.mm(r_f_norm.t(), s_f_norm)
    
    # 只惩罚对角线元素（自相关）
    rs_ss_diag = torch.diag(rs_ss_corr).pow(2).mean()
    rf_sf_diag = torch.diag(rf_sf_corr).pow(2).mean()
    
    return rs_ss_diag + rf_sf_diag



def cluster_loss_function(q, p=None):
    """恢复原始聚类损失函数"""
    if p is None:
        p = target_distribution(q, sharpen=1.5)
    
    # 原始简单版本
    q_log = torch.log(q + 1e-8)
    kl_loss = F.kl_div(q_log, p, reduction='batchmean')
    return kl_loss, p


def train_epoch(model, optimizer, features, A_spatial, A_feature, graph_nei, graph_neg, config, p_target=None, update_graph=False):
    """训练一个epoch - 支持动态图更新"""
    model.train()
    train_epoch.domain_disc.train()  # 设置域判别器为训练模式
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(features, A_spatial, features, A_feature)
    
    Z = outputs['Z']
    q = outputs['q']
    pi = outputs['pi']
    disp = outputs['disp']
    mean = outputs['mean']
    A_s_hat = outputs['A_s_hat']
    A_f_hat = outputs['A_f_hat']  # 新增：特征图重建
    z_f = outputs['z_f']
    z_s = outputs['z_s']
    r_f = outputs['r_f']
    r_s = outputs['r_s']
    s_f = outputs['s_f']
    s_s = outputs['s_s']
    
    # 计算各项损失
    loss_expr = zinb_loss_function(features, pi, disp, mean)
    loss_struct_spatial = structure_loss_function(A_spatial, A_s_hat)
    loss_struct_feature = structure_loss_function(A_feature, A_f_hat)  # 新增：特征图重建损失
    
    # 聚类损失 - 关键修改
    if p_target is not None:
        loss_cluster, _ = cluster_loss_function(q, p_target)
    else:
        loss_cluster, p_new = cluster_loss_function(q)
        p_target = p_new
    
    reg_loss_f = regularization_loss(z_f, graph_nei, graph_neg)
    reg_loss_s = regularization_loss(z_s, graph_nei, graph_neg)
    reg_loss = (reg_loss_f + reg_loss_s) / 2
    
    dicr_loss_val = dicr_loss(z_f, z_s)
    
    # 新增：域判别与正交约束损失
    dom_loss = domain_ce_loss(train_epoch.domain_disc, s_s, s_f)
    loss_orth = orthogonality_loss(r_s, s_s, r_f, s_f)
    
    # 总损失 - 加入特征图重建损失和两个关键损失
    total_loss = (config.alpha * loss_expr + 
                  config.beta * loss_struct_spatial + 
                  config.theta * loss_struct_feature +  # 新增：特征图重建损失
                  config.gamma * loss_cluster + 
                  config.delta * reg_loss + 
                  config.dicr_weight * dicr_loss_val +
                  config.dom_weight * dom_loss +
                  config.orth_weight * loss_orth)
    
    # 反向传播
    total_loss.backward()
    # 梯度裁剪防止梯度爆炸（恢复原始配置）
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(train_epoch.domain_disc.parameters()), max_norm=1.0)
    optimizer.step()
    
    # 返回结果
    Z_np = pd.DataFrame(Z.detach().cpu().tolist()).fillna(0).values
    mean_np = pd.DataFrame(mean.detach().cpu().tolist()).fillna(0).values
    
    # 如果需要更新图，返回重建的mean用于图更新
    if update_graph:
        return Z_np, mean_np, loss_expr, loss_struct_spatial, loss_struct_feature, loss_cluster, reg_loss, dicr_loss_val, total_loss, p_target, mean
    else:
        return Z_np, mean_np, loss_expr, loss_struct_spatial, loss_struct_feature, loss_cluster, reg_loss, dicr_loss_val, total_loss, p_target


def evaluate_clustering(Z, labels, n_clusters):
    """与quick测试保持一致的聚类评估"""
    # 处理Z矩阵，与quick测试保持一致
    if isinstance(Z, torch.Tensor):
        Z_np = pd.DataFrame(Z.detach().cpu().tolist()).fillna(0).values
    else:
        Z_np = Z
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit(Z_np)
    idx = kmeans.labels_
    ari_res = metrics.adjusted_rand_score(labels, idx)
    nmi_res = metrics.normalized_mutual_info_score(labels, idx)
    return ari_res, nmi_res, idx


def plot_training_curves(loss_history, ari_history, savepath, dataset):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{dataset} Training Progress', fontsize=16, fontweight='bold')
    
    # 损失曲线
    axes[0, 0].plot(loss_history['total'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 各项损失分解
    axes[0, 1].plot(loss_history['expr'], 'r-', linewidth=2, label='Expression Loss')
    axes[0, 1].plot(loss_history['struct_spatial'], 'g-', linewidth=2, label='Spatial Structure Loss')
    axes[0, 1].plot(loss_history['struct_feature'], 'purple', linewidth=2, label='Feature Structure Loss')
    axes[0, 1].plot(loss_history['cluster'], 'm-', linewidth=2, label='Cluster Loss')
    axes[0, 1].set_title('Loss Components', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 正则化和DICR损失
    axes[0, 2].plot(loss_history['reg'], 'c-', linewidth=2, label='Regularization Loss')
    axes[0, 2].plot(loss_history['dicr'], 'orange', linewidth=2, label='DICR Loss')
    axes[0, 2].set_title('Regularization Losses', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # ARI曲线
    axes[1, 0].plot(ari_history, 'purple', linewidth=3, marker='o', markersize=3)
    axes[1, 0].set_title('ARI Score Progress', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ARI Score')
    axes[1, 0].grid(True, alpha=0.3)
    max_ari = max(ari_history)
    max_epoch = ari_history.index(max_ari)
    axes[1, 0].axhline(y=max_ari, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].text(0.02, 0.98, f'Max ARI: {max_ari:.4f}\nEpoch: {max_epoch}', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    # 图更新标记
    axes[1, 1].plot(ari_history, 'purple', linewidth=2, alpha=0.7)
    graph_update_epochs = [i for i in range(len(ari_history)) if i > 10 and i % 20 == 0]
    for epoch in graph_update_epochs:
        if epoch < len(ari_history):
            axes[1, 1].axvline(x=epoch, color='red', linestyle=':', alpha=0.6)
            axes[1, 1].scatter(epoch, ari_history[epoch], color='red', s=50, zorder=5)
    axes[1, 1].set_title('ARI with Graph Update Points', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ARI Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 损失比例饼图
    final_losses = {
        'Expression': loss_history['expr'][-1] if loss_history['expr'] else 0,
        'Spatial Structure': loss_history['struct_spatial'][-1] if loss_history['struct_spatial'] else 0,
        'Feature Structure': loss_history['struct_feature'][-1] if loss_history['struct_feature'] else 0,
        'Cluster': loss_history['cluster'][-1] if loss_history['cluster'] else 0,
        'Regularization': loss_history['reg'][-1] if loss_history['reg'] else 0,
        'DICR': loss_history['dicr'][-1] if loss_history['dicr'] else 0
    }
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    axes[1, 2].pie(final_losses.values(), labels=final_losses.keys(), autopct='%1.1f%%',
                   colors=colors, startangle=90)
    axes[1, 2].set_title('Final Loss Composition', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{savepath}training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_embeddings_2d(embeddings, labels, pred_labels, spatial_coords, savepath, dataset, epoch=None):
    """绘制2D嵌入可视化"""
    # 使用t-SNE和PCA降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    pca = PCA(n_components=2, random_state=42)
    
    emb_tsne = tsne.fit_transform(embeddings)
    emb_pca = pca.fit_transform(embeddings)
    
    # 将标签转换为数字类型以避免类型不匹配
    if isinstance(labels[0], str):
        # 如果labels是字符串，转换为数字
        unique_labels = np.unique(labels)
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        labels_numeric = np.array([label_to_num[label] for label in labels])
    else:
        labels_numeric = np.array(labels)
    
    # 确保pred_labels也是numpy数组
    pred_labels_numeric = np.array(pred_labels)
    
    # 创建颜色映射（使用自定义颜色，簇数>5时循环使用）
    n_clusters = len(np.unique(labels_numeric))
    custom_colors = np.array(['#3473f8', '#32e3da', '#99fca6', '#ffb360', '#ff2613'])
    colors = custom_colors[np.arange(n_clusters) % len(custom_colors)]
    
    # 统一圆点输出规格
    point_size = 12
    point_alpha = 0.9
    edge_color = 'black'
    edge_width = 0.2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    epoch_str = f'Epoch {epoch}' if epoch is not None else 'Final'
    fig.suptitle(f'{dataset} Embeddings Visualization - {epoch_str}', fontsize=16, fontweight='bold')
    
    # t-SNE - 真实标签
    for i, label in enumerate(np.unique(labels_numeric)):
        mask = labels_numeric == label
        axes[0, 0].scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                           c=[colors[i]], label=f'Cluster {label}',
                           alpha=point_alpha, s=point_size,
                           edgecolors=edge_color, linewidths=edge_width)
    axes[0, 0].set_title('t-SNE - Ground Truth', fontweight='bold')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # t-SNE - 预测标签
    for i, label in enumerate(np.unique(pred_labels_numeric)):
        mask = pred_labels_numeric == label
        axes[0, 1].scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                           c=[colors[i]], label=f'Pred {label}',
                           alpha=point_alpha, s=point_size,
                           edgecolors=edge_color, linewidths=edge_width)
    axes[0, 1].set_title('t-SNE - Predictions', fontweight='bold')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    
    # PCA - 真实标签
    for i, label in enumerate(np.unique(labels_numeric)):
        mask = labels_numeric == label
        axes[0, 2].scatter(emb_pca[mask, 0], emb_pca[mask, 1],
                           c=[colors[i]], label=f'Cluster {label}',
                           alpha=point_alpha, s=point_size,
                           edgecolors=edge_color, linewidths=edge_width)
    axes[0, 2].set_title('PCA - Ground Truth', fontweight='bold')
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 空间分布 - 真实标签
    if spatial_coords is not None:
        for i, label in enumerate(np.unique(labels_numeric)):
            mask = labels_numeric == label
            axes[1, 0].scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                               c=[colors[i]], label=f'Cluster {label}',
                               alpha=point_alpha, s=point_size,
                               edgecolors=edge_color, linewidths=edge_width)
        axes[1, 0].set_title('Spatial Distribution - Ground Truth', fontweight='bold')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].set_xlabel('X coordinate')
        axes[1, 0].set_ylabel('Y coordinate')
        
        # 空间分布 - 预测标签
        for i, label in enumerate(np.unique(pred_labels_numeric)):
            mask = pred_labels_numeric == label
            axes[1, 1].scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                               c=[colors[i]], label=f'Pred {label}',
                               alpha=point_alpha, s=point_size,
                               edgecolors=edge_color, linewidths=edge_width)
        axes[1, 1].set_title('Spatial Distribution - Predictions', fontweight='bold')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].set_xlabel('X coordinate')
        axes[1, 1].set_ylabel('Y coordinate')
    
    # 混淆矩阵热图 - 使用数字标签
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels_numeric, pred_labels_numeric)
    im = axes[1, 2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 2].set_title('Confusion Matrix', fontweight='bold')
    axes[1, 2].set_xlabel('Predicted Label')
    axes[1, 2].set_ylabel('True Label')
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 2].text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    epoch_suffix = f'_epoch_{epoch}' if epoch is not None else '_final'
    plt.savefig(f'{savepath}embeddings_2d{epoch_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 单独输出聚类图（仅预测标签，t-SNE / PCA / Spatial）
    def save_single_scatter(xy, labels_arr, fname, title):
        plt.figure(figsize=(5, 4))
        for i, label in enumerate(np.unique(labels_arr)):
            mask = labels_arr == label
            plt.scatter(xy[mask, 0], xy[mask, 1], s=point_size, c=[colors[i]],
                        alpha=point_alpha, edgecolors=edge_color, linewidths=edge_width, label=str(label))
        plt.title(title)
        plt.legend(fontsize=7, markerscale=1, frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, fname), dpi=300, bbox_inches='tight')
        plt.close()

    save_single_scatter(emb_tsne, pred_labels_numeric, 'tsne_pred.png', 't-SNE (Predicted Clusters)')
    save_single_scatter(emb_pca, pred_labels_numeric, 'pca_pred.png', 'PCA (Predicted Clusters)')
    if spatial_coords is not None:
        save_single_scatter(spatial_coords, pred_labels_numeric, 'spatial_pred.png', 'Spatial (Predicted Clusters)')


def plot_graph_analysis(A_spatial, A_feature_original, A_feature_updated, savepath, dataset):
    """分析和可视化图结构变化"""
    # 确保输入为 numpy 数组，避免 list 上的算术报错
    if isinstance(A_feature_original, list):
        A_feature_original = np.array(A_feature_original)
    if isinstance(A_feature_updated, list):
        A_feature_updated = np.array(A_feature_updated)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset} Graph Structure Analysis', fontsize=16, fontweight='bold')
    
    # 度分布比较
    degrees_spatial = np.sum(A_spatial.cpu().tolist() if torch.is_tensor(A_spatial) else A_spatial, axis=1)
    degrees_feature_orig = np.sum(A_feature_original, axis=1)
    degrees_feature_updated = np.sum(A_feature_updated, axis=1)
    
    axes[0, 0].hist(degrees_spatial, bins=20, alpha=0.7, label='Spatial Graph', color='blue')
    axes[0, 0].hist(degrees_feature_orig, bins=20, alpha=0.7, label='Original Feature Graph', color='red')
    axes[0, 0].hist(degrees_feature_updated, bins=20, alpha=0.7, label='Updated Feature Graph', color='green')
    axes[0, 0].set_title('Degree Distribution Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Node Degree')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图密度比较
    n_nodes = A_spatial.shape[0]
    max_edges = n_nodes * (n_nodes - 1) / 2
    
    density_spatial = np.sum(A_spatial.cpu().tolist() if torch.is_tensor(A_spatial) else A_spatial) / 2 / max_edges
    density_feature_orig = np.sum(A_feature_original) / 2 / max_edges
    density_feature_updated = np.sum(A_feature_updated) / 2 / max_edges
    
    densities = [density_spatial, density_feature_orig, density_feature_updated]
    labels = ['Spatial', 'Feature (Orig)', 'Feature (Updated)']
    colors = ['blue', 'red', 'green']
    
    bars = axes[0, 1].bar(labels, densities, color=colors, alpha=0.7)
    axes[0, 1].set_title('Graph Density Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标注
    for bar, density in zip(bars, densities):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{density:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 边的变化分析 - 修复负值问题
    edge_changes = A_feature_updated - A_feature_original
    
    # 计算边的变化统计，确保都是非负值
    added_edges = np.sum(edge_changes > 0) / 2  # 新增的边
    removed_edges = np.sum(edge_changes < 0) / 2  # 删除的边
    unchanged_edges = np.sum((A_feature_original == 1) & (A_feature_updated == 1)) / 2  # 保持不变的边
    
    # 确保所有值都是非负的
    added_edges = max(0, added_edges)
    removed_edges = max(0, removed_edges)
    unchanged_edges = max(0, unchanged_edges)
    
    # 如果所有值都为0，设置默认值避免空饼图
    if added_edges == 0 and removed_edges == 0 and unchanged_edges == 0:
        unchanged_edges = 1
        change_labels = ['No Changes']
        change_data = [1]
        change_colors = ['gray']
    else:
        change_data = [unchanged_edges, added_edges, removed_edges]
        change_labels = [f'Unchanged ({int(unchanged_edges)})', 
                        f'Added ({int(added_edges)})', 
                        f'Removed ({int(removed_edges)})']
        change_colors = ['gray', 'green', 'red']
        
        # 过滤掉值为0的项
        filtered_data = []
        filtered_labels = []
        filtered_colors = []
        for data, label, color in zip(change_data, change_labels, change_colors):
            if data > 0:
                filtered_data.append(data)
                filtered_labels.append(label)
                filtered_colors.append(color)
        
        change_data = filtered_data
        change_labels = filtered_labels
        change_colors = filtered_colors
    
    axes[0, 2].pie(change_data, labels=change_labels, autopct='%1.1f%%',
                   colors=change_colors, startangle=90)
    axes[0, 2].set_title('Edge Changes Distribution', fontweight='bold')
    
    # 邻接矩阵可视化（采样显示）
    sample_size = min(100, n_nodes)  # 最多显示100个节点
    indices = np.random.choice(n_nodes, sample_size, replace=False)
    indices = np.sort(indices)
    
    A_spatial_sample = np.array(A_spatial.cpu().tolist())[np.ix_(indices, indices)] if torch.is_tensor(A_spatial) else A_spatial[np.ix_(indices, indices)]
    A_feature_orig_sample = A_feature_original[np.ix_(indices, indices)]
    A_feature_updated_sample = A_feature_updated[np.ix_(indices, indices)]
    
    im1 = axes[1, 0].imshow(A_spatial_sample, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Spatial Graph (Sample)', fontweight='bold')
    axes[1, 0].set_xlabel('Node Index')
    axes[1, 0].set_ylabel('Node Index')
    
    im2 = axes[1, 1].imshow(A_feature_orig_sample, cmap='Reds', aspect='auto')
    axes[1, 1].set_title('Original Feature Graph (Sample)', fontweight='bold')
    axes[1, 1].set_xlabel('Node Index')
    axes[1, 1].set_ylabel('Node Index')
    
    im3 = axes[1, 2].imshow(A_feature_updated_sample, cmap='Greens', aspect='auto')
    axes[1, 2].set_title('Updated Feature Graph (Sample)', fontweight='bold')
    axes[1, 2].set_xlabel('Node Index')
    axes[1, 2].set_ylabel('Node Index')
    
    plt.tight_layout()
    plt.savefig(f'{savepath}graph_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(dataset, config, ari_max, nmi_max, epoch_max, loss_history, savepath):
    """创建训练总结报告"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset} Training Summary Report', fontsize=18, fontweight='bold')
    
    # 主要指标展示
    metrics_text = f"""
    Dataset: {dataset}
    
    Final Results:
    • Best ARI: {ari_max:.4f}
    • Best NMI: {nmi_max:.4f}
    • Best Epoch: {epoch_max}
    
    Model Configuration:
    • Learning Rate: {config.lr}
    • Epochs: {config.epochs}
    • Hidden Dims: {config.nhid1} → {config.nhid2}
    • Projection Dim: {config.proj_dim}
    
    Loss Weights:
    • α (Expression): {config.alpha}
    • β (Spatial Structure): {config.beta}
    • γ (Clustering): {config.gamma}
    • δ (Regularization): {config.delta}
    • DICR Weight: {config.dicr_weight}
    • θ (Feature Structure): {config.theta}
    
    Graph Update:
    • Update Interval: {config.graph_update_interval} epochs
    • Target Update: {config.update_interval} epochs
    """
    
    axes[0, 0].text(0.05, 0.95, metrics_text, transform=axes[0, 0].transAxes,
                    verticalalignment='top', fontsize=11, fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Configuration & Results', fontweight='bold', fontsize=14)
    
    # 损失趋势
    epochs = range(len(loss_history['total']))
    axes[0, 1].plot(epochs, loss_history['total'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 1].set_title('Training Loss Curve', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 性能提升分析
    if len(loss_history['total']) > 50:
        early_loss = np.mean(loss_history['total'][:50])
        late_loss = np.mean(loss_history['total'][-50:])
        improvement = (early_loss - late_loss) / early_loss * 100
        
        improvement_text = f"""
        Performance Analysis:
        
        • Early Training Loss (first 50 epochs): {early_loss:.4f}
        • Late Training Loss (last 50 epochs): {late_loss:.4f}
        • Loss Improvement: {improvement:.2f}%
        
        Graph Updates Performed:
        • Total Updates: {len([i for i in range(config.epochs) if i > 10 and i % config.graph_update_interval == 0])}
        • Update Frequency: Every {config.graph_update_interval} epochs
        
        Training Phases:
        • Phase 1: Pre-training (10 epochs)
        • Phase 2: Full training with dynamic graphs
        """
        
        axes[1, 0].text(0.05, 0.95, improvement_text, transform=axes[1, 0].transAxes,
                        verticalalignment='top', fontsize=11, fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Performance Analysis', fontweight='bold', fontsize=14)
    
    # 损失组成分析
    final_losses = {
        'Expression': loss_history['expr'][-1] if loss_history['expr'] else 0,
        'Spatial Structure': loss_history['struct_spatial'][-1] if loss_history['struct_spatial'] else 0,
        'Feature Structure': loss_history['struct_feature'][-1] if loss_history['struct_feature'] else 0,
        'Cluster': loss_history['cluster'][-1] if loss_history['cluster'] else 0,
        'Regularization': loss_history['reg'][-1] if loss_history['reg'] else 0,
        'DICR': loss_history['dicr'][-1] if loss_history['dicr'] else 0
    }
    
    colors = ['#8000ff', '#1996f3', '#4df3ce', '#b3f396', '#ff964f']
    wedges, texts, autotexts = axes[1, 1].pie(final_losses.values(), labels=final_losses.keys(), 
                                              autopct='%1.2f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Final Loss Composition', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{savepath}summary_report.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 生成时间戳用于区分不同的实验
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    datasets = ['151672' ]
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # 加载数据
        features, labels, A_spatial, A_feature, graph_nei, graph_neg = load_data(dataset)
        
        # 配置参数
        config = SimpleConfig()
        cuda = not config.no_cuda and torch.cuda.is_available()
        
        # 处理标签
        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())
        
        print(f"Number of clusters: {config.class_num}")

        if cuda:
            features = features.cuda()
            A_spatial = A_spatial.cuda()
            A_feature = A_feature.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        # 设置随机种子（简化版，与quick测试一致）
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)
        # 确保结果尽量可复现（避免触发 CuBLAS 非确定性报错）
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 不强制 use_deterministic_algorithms(True)，以避免 CuBLAS 错误；
            # 如需严格确定性，请在外部设置 CUBLAS_WORKSPACE_CONFIG 并自行开启。
        except Exception:
            pass

        print(f'{dataset} lr={config.lr} alpha={config.alpha} beta={config.beta} gamma={config.gamma}')
        
        # 初始化新模型
        model = SharedSpecificNet(
            in_dim=features.shape[1],
            hid_dim=config.nhid1,
            out_dim=config.nhid2,
            proj_dim=config.proj_dim,
            n_clusters=config.class_num,
            dropout=config.dropout
        )
        
        # 初始化域判别器并绑定到train_epoch
        domain_disc = DomainDiscriminator(config.nhid2)  # 使用hid_dim作为输入维度
        train_epoch.domain_disc = domain_disc.cuda() if cuda else domain_disc
        
        if cuda:
            model.cuda()
            
        # 同时优化模型和域判别器参数
        optimizer = optim.Adam(list(model.parameters()) + list(train_epoch.domain_disc.parameters()), 
                              lr=config.lr, weight_decay=config.weight_decay)
        
        # 移除学习率调度器，与quick测试保持一致
        scheduler = None

        # 获取空间坐标用于可视化
        path = "./generate_data/DLPFC/" + dataset + "/MAFN.h5ad"
        adata = sc.read_h5ad(path)
        if 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']
        
        # 创建带时间戳的保存路径 - 为每个切片创建独立文件夹
        savepath = f'./result/DLPFC/{dataset}_{timestamp}/'
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)
        
        print(f"Results will be saved to: {savepath}")
        
        # 保存实验配置信息
        config_info = {
            'timestamp': timestamp,
            'dataset': dataset,
            'model_config': {
                'epochs': config.epochs,
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'nhid1': config.nhid1,
                'nhid2': config.nhid2,
                'proj_dim': config.proj_dim,
                'dropout': config.dropout,
                'k': config.k,
                'radius': config.radius
            },
            'loss_weights': {
                'alpha': config.alpha,
                'beta': config.beta,
                'gamma': config.gamma,
                'delta': config.delta,
                'dicr_weight': config.dicr_weight,
                'theta': config.theta
            },
            'training_params': {
                'update_interval': config.update_interval,
                'graph_update_interval': config.graph_update_interval,
                'seed': config.seed
            },
            'data_info': {
                'n_spots': features.shape[0],
                'n_features': features.shape[1],
                'n_clusters': config.class_num
            }
        }
        
        # 保存配置到JSON文件
        with open(f'{savepath}experiment_config.json', 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
        
        # 训练循环 - 关键改进
        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        best_nmi_at_best_ari = 0.0  # 记录在最佳 ARI 时对应的 NMI
        p_target = None  # 目标分布
        
        # 初始化历史记录用于可视化
        loss_history = {
            'total': [],
            'expr': [],
            'struct_spatial': [],
            'struct_feature': [],
            'cluster': [],
            'reg': [],
            'dicr': []
        }
        ari_history = []
        A_feature_original = A_feature.cpu().tolist() if cuda else A_feature.tolist()
        A_feature_updated = A_feature_original.copy()

        # 移除预训练阶段，与quick测试保持一致
        print("Starting full training...")
        # 正式训练阶段
        for epoch in range(config.epochs):
            # 每个epoch都更新目标分布，与quick测试保持一致
            model.eval()
            with torch.no_grad():
                outputs = model(features, A_spatial, features, A_feature)
                q_current = outputs['q']
                p_target = target_distribution(q_current)
                if cuda:
                    p_target = p_target.cuda()
            
            # 基线：禁用动态图更新（与最初状态一致）
            update_graph_flag = False
            
            if update_graph_flag:
                # 需要更新图的训练
                result = train_epoch(
                    model, optimizer, features, A_spatial, A_feature, graph_nei, graph_neg, 
                    config, p_target, update_graph=True
                )
                Z, mean, loss_expr, loss_struct_spatial, loss_struct_feature, loss_cluster, reg_loss, dicr_loss_val, total_loss, _, mean_tensor = result
                
                # 使用ZINB重建的mean更新特征图
                print(f"Epoch {epoch}: Updating feature graph with ZINB reconstructed mean...")
                A_feature_new, A_feature_raw = construct_feature_graph_from_mean(
                    mean_tensor, k=config.k, cuda=cuda
                )
                A_feature = A_feature.to_dense()

                # 保存更新后的特征图用于可视化
                A_feature_updated = A_feature_raw.copy()  # 注意：变量名要和上面构图时一致！

                # ✅ 加权融合表达图 + 空间图，构建正样本图（针对151672优化）
                alpha = 0.7  # 空间图占 70%（增加空间信息权重）
                graph_nei_new = alpha * np.array(A_spatial.cpu().tolist()) + (1 - alpha) * A_feature_raw
                graph_nei_new = (graph_nei_new > 0).astype(float)
                graph_nei_new = torch.FloatTensor(graph_nei_new)

                # ✅ 负样本图 = 全 1 - 正样本图
                graph_neg_new = torch.ones_like(graph_nei_new) - graph_nei_new

                # ✅ CUDA 迁移
                if cuda:
                    graph_nei_new = graph_nei_new.cuda()
                    graph_neg_new = graph_neg_new.cuda()

                graph_nei = graph_nei_new
                graph_neg = graph_neg_new

                print(f"[加权融合] Graph updated! alpha={alpha} → total edges: {torch.sum(graph_nei)}")

            else:
                # 正常训练
                Z, mean, loss_expr, loss_struct_spatial, loss_struct_feature, loss_cluster, reg_loss, dicr_loss_val, total_loss, _ = train_epoch(
                    model, optimizer, features, A_spatial, A_feature, graph_nei, graph_neg, config, p_target
                )
            
            # 记录损失历史
            loss_history['total'].append(total_loss.item())
            loss_history['expr'].append(loss_expr.item())
            loss_history['struct_spatial'].append(loss_struct_spatial.item())
            loss_history['struct_feature'].append(loss_struct_feature.item())
            loss_history['cluster'].append(loss_cluster.item())
            loss_history['reg'].append(reg_loss.item())
            loss_history['dicr'].append(dicr_loss_val.item())
            
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f'{dataset} epoch: {epoch} '
                      f'expr_loss={loss_expr:.2f} '
                      f'struct_spatial_loss={loss_struct_spatial:.2f} '
                      f'struct_feature_loss={loss_struct_feature:.2f} '
                      f'cluster_loss={loss_cluster:.2f} '
                      f'reg_loss={reg_loss:.2f} '
                      f'dicr_loss={dicr_loss_val:.2f} '
                      f'total_loss={total_loss:.2f}')
            
            # 聚类评估（提高频率：每5个epoch评估一次）
            if epoch % 5 == 0:
                ari_res, nmi_res, idx = evaluate_clustering(Z, labels, config.class_num)
                ari_history.append(ari_res)
            else:
                # 非评估epoch，使用上一次的 ARI；NMI 保持上一次评估值
                if len(ari_history) > 0:
                    ari_res = ari_history[-1]
                    idx = None
                else:
                    ari_res, nmi_res, idx = evaluate_clustering(Z, labels, config.class_num)
                    ari_history.append(ari_res)
            
            # 移除学习率调度，与quick测试保持一致
            # scheduler.step(ari_res)
            
            # 中间结果可视化保存
            if config.save_plots and epoch % config.plot_interval == 0 and epoch > 0:
                print(f"Saving intermediate visualization at epoch {epoch}...")
                plot_embeddings_2d(Z, labels, idx, spatial_coords, savepath, dataset, epoch)
            
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = Z
                # 记录该时刻的 NMI
                try:
                    best_nmi_at_best_ari = float(nmi_res)
                except Exception:
                    pass

        print(f'{dataset} Best ARI: {ari_max:.4f}, NMI = {best_nmi_at_best_ari:.4f} at epoch {epoch_max}')

        # 保存详细的实验结果
        results_summary = {
            'final_metrics': {
                'best_ari': float(ari_max),
                'best_nmi': float(nmi_res),
                'best_epoch': int(epoch_max)
            },
            'training_history': {
                'ari_history': [float(x) for x in ari_history],
                'loss_history': {k: [float(x) for x in v] for k, v in loss_history.items()}
            },
            'experiment_info': config_info
        }
        
        # 保存实验结果摘要
        with open(f'{savepath}results_summary.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)

        # ===== 可视化输出（按用户提供格式） =====
        try:
            # 重新读取原始adata
            ad_path = "./generate_data/DLPFC/" + dataset + "/MAFN.h5ad"
            adata = sc.read_h5ad(ad_path)

            # 1) 手动标注（粗/细）可视化
            plt.rcParams["figure.figsize"] = (3, 3)
            title_coarse = "Manual annotation (slice #" + dataset + ")"
            if 'ground_truth' in adata.obs.columns:
                sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title_coarse, show=False)
                plt.savefig(savepath + 'Manual_Annotation_coarse.jpg', bbox_inches='tight', dpi=600)
                plt.close()
            if 'fine_annot_type' in adata.obs.columns:
                sc.pl.spatial(adata, img_key="hires", color=['fine_annot_type'], title="Fine annotation (slice #" + dataset + ")", show=False)
                plt.savefig(savepath + 'Manual_Annotation_fine.jpg', bbox_inches='tight', dpi=600)
                plt.close()

            # 2) 训练得到的聚类结果可视化（白底、不叠加原图）
            # 与每50轮输出保持一致的配色（5色循环）
            base_colors = ['#3473f8', '#7f00ff', '#99fca6', '#ffb360', '#ff2613']
            title_pred = f'ARI={ari_max:.2f}'
            if isinstance(idx_max, (list, np.ndarray)) and len(idx_max) == adata.n_obs:
                idx_arr = np.array(idx_max)
                # 保持颜色映射顺序：按数值标签排序（np.unique 默认升序）
                unique_order = np.unique(idx_arr)
                palette = [base_colors[i % len(base_colors)] for i in unique_order]
                # 写入为分类类型并固定类别顺序
                adata.obs['idx'] = idx_arr.astype(str)
                # 使用pandas分类以固定类别顺序
                adata.obs['idx'] = pd.Categorical(adata.obs['idx'],
                                                  categories=[str(x) for x in unique_order],
                                                  ordered=True)
            if isinstance(emb_max, (list, np.ndarray)):
                adata.obsm['emb'] = np.array(emb_max)
            if isinstance(mean_max, (list, np.ndarray)):
                adata.obsm['mean'] = np.array(mean_max)

            sc.pl.spatial(adata, color=['idx'], title=title_pred, show=False,
                          palette=palette, img_key=None, frameon=False, size=1.2)
            plt.gca().set_facecolor('white')
            plt.gcf().patch.set_facecolor('white')
            plt.savefig(savepath + 'MAFN.jpg', bbox_inches='tight', dpi=600, facecolor='white', edgecolor='none')
            plt.close()

            # 3) 基于mean的UMAP + PAGA对比
            if 'mean' in adata.obsm:
                sc.pp.neighbors(adata, use_rep='mean')
                sc.tl.umap(adata)
                sc.tl.paga(adata, groups='idx')
                sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title_pred,
                                   legend_fontoutline=2, show=False)
                plt.savefig(savepath + 'MAFN_umap_mean.jpg', bbox_inches='tight', dpi=600)
                plt.close()

            # 4) 保存嵌入与标签、写回h5ad
            try:
                pd.DataFrame(emb_max).to_csv(savepath + 'MAFN_emb.csv', index=False)
                pd.DataFrame(idx_max).to_csv(savepath + 'MAFN_idx.csv', index=False)
            except Exception:
                pass
            adata.layers['X'] = adata.X
            if isinstance(mean_max, (list, np.ndarray)):
                adata.layers['mean'] = np.array(mean_max)
            adata.write(savepath + 'MAFN.h5ad')

            # 5) 仅UMAP聚类着色
            if 'X_umap' in adata.obsm and 'idx' in adata.obs.columns:
                sc.pl.umap(adata, color=['idx'], frameon=False, show=False)
                plt.savefig(savepath + 'MAFN_umap_idx.jpg', bbox_inches='tight', dpi=600)
                plt.close()
        except Exception as e:
            print(f'[Visualization] skipped due to error: {e}')

        # 保存结果文件
        pd.DataFrame(emb_max).to_csv(savepath + 'SharedSpecific_emb.csv', index=False)
        pd.DataFrame(idx_max).to_csv(savepath + 'SharedSpecific_idx.csv', index=False)
        pd.DataFrame(mean_max).to_csv(savepath + 'SharedSpecific_mean.csv', index=False)
        
        # 保存训练历史
        pd.DataFrame(loss_history).to_csv(savepath + 'training_loss_history.csv', index=False)
        pd.DataFrame({'epoch': range(len(ari_history)), 'ari': ari_history}).to_csv(savepath + 'ari_history.csv', index=False)
        
        print(f"Results saved to {savepath}")
        print(f"  - Embeddings: SharedSpecific_emb.csv")
        print(f"  - Cluster assignments: SharedSpecific_idx.csv")
        print(f"  - Reconstructed expression: SharedSpecific_mean.csv")
        print(f"  - Configuration: experiment_config.json")
        print(f"  - Results summary: results_summary.json")
        print(f"  - Training history: training_loss_history.csv, ari_history.csv")
        
        # 生成所有可视化图表
        print("Generating visualizations...")
        
        # 绘制训练曲线
        plot_training_curves(loss_history, ari_history, savepath, dataset)
        print("Training curves saved")

        # 绘制最终嵌入可视化
        plot_embeddings_2d(emb_max, labels, idx_max, spatial_coords, savepath, dataset)
        print("Final embeddings visualization saved")

        # 绘制图结构分析
        plot_graph_analysis(A_spatial, A_feature_original, A_feature_updated, savepath, dataset)
        print("✓ Graph analysis saved")

        # 创建训练总结报告
        create_summary_report(dataset, config, ari_max, nmi_res, epoch_max, loss_history, savepath)
        print("✓ Summary report saved")
        
        print(f"All visualizations saved to {savepath}")
        print("="*60)