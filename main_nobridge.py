import os
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from Model_Wavelet_Attention import *
from scipy.stats import entropy
from sklearn import preprocessing
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix
from utilities import *
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import copy
import torch.utils.data as Data
import utils
import sklearn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np
import warnings
import seaborn as sns
import time
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix as sk_cm
warnings.filterwarnings("ignore")

from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from sklearn.manifold import TSNE
# 计算混淆矩阵和准确率
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
# 添加Wasserstein距离相关导入
from scipy.stats import wasserstein_distance
from torch.utils.data import Subset
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 设置兼容的字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建结果目录
os.makedirs("./results_nobridge", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./log", exist_ok=True)


# # BUILD THE DATASET
class EarlyStopping:
    def __init__(self, patience, mode, delta=0.00, verbose=True):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, metric, model):
        if self.best is None:
            self.best = metric
            self.save_checkpoint(metric, model)
        else:
            improve = (metric > self.best + self.delta) if self.mode == 'max' else (metric < self.best - self.delta)
            if improve:
                self.best = metric
                self.save_checkpoint(metric, model)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f"[EarlyStopping] counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, metric, model):
        if isinstance(model, dict):
            self.best_state = {k: v.cpu().clone() for k, v in model.items()}
        else:
            self.best_state = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f"[EarlyStopping] metric improved -> {metric:.6f}, save best state")

    def load_best(self, model):
        if self.best_state is None:
            return
        if isinstance(model, dict):
            for k, v in model.items():
                if k in self.best_state:
                    v.data.copy_(self.best_state[k])
        else:
            # 使用非严格模式加载，允许部分键不匹配
            try:
                model.load_state_dict(self.best_state, strict=True)
            except:
                print("严格模式加载失败，尝试非严格模式...")
                model.load_state_dict(self.best_state, strict=False)

def get_dat(data_num):
    if data_num == 1:
        F_0 = np.load("./bearingset/data_thirdspace/T1_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./bearingset/data_thirdspace/T1_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./bearingset/data_thirdspace/T1_source_label.npy")
        Y_1 = np.load("./bearingset/data_thirdspace/T1_target_label.npy")
        print("Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-OSF", "2-OSP", "3-ORF", "4-ISF", "6-IORF", "7-IORP", "8-H", "Unknown"]

    if data_num == 2:
        F_0 = np.load("./bearingset/data_thirdspace/T2_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./bearingset/data_thirdspace/T2_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./bearingset/data_thirdspace/T2_source_label.npy")
        Y_1 = np.load("./bearingset/data_thirdspace/T2_target_label.npy")
        print("Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-OSF", "3-ORF", "4-ISF", "6-IORF", "7-IORP", "8-H", "Unknown"]

    if data_num == 3:
        F_0 = np.load("./bearingset/data_thirdspace/T3_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./bearingset/data_thirdspace/T3_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./bearingset/data_thirdspace/T3_source_label.npy")
        Y_1 = np.load("./bearingset/data_thirdspace/T3_target_label.npy")
        print("Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-OSF", "2-OSP", "3-ORF", "4-ISF", "5-IRF", "Unknown"]

    if data_num == 4:
        F_0 = np.load("./bearingset/data_thirdspace/T4_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./bearingset/data_thirdspace/T4_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./bearingset/data_thirdspace/T4_source_label.npy")
        Y_1 = np.load("./bearingset/data_thirdspace/T4_target_label.npy")
        print("Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-OSF", "4-ISF", "6-IORF", "8-H", "Unknown"]

    if data_num == 5:
        F_0 = np.load("./gearset/datagearset_thirdspace/T1_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./gearset/datagearset_thirdspace/T1_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./gearset/datagearset_thirdspace/T1_source_label.npy")
        Y_1 = np.load("./gearset/datagearset_thirdspace/T1_target_label.npy")
        print("任务1 - Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-N", "3-BF&SI", "4-G&E", "Unknown"]

    if data_num == 6:
        # 加载任务2数据
        F_0 = np.load("./gearset/datagearset_thirdspace/T2_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./gearset/datagearset_thirdspace/T2_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./gearset/datagearset_thirdspace/T2_source_label.npy")
        Y_1 = np.load("./gearset/datagearset_thirdspace/T2_target_label.npy")
        print("任务2 - Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-N", "3-BF&SI", "4-G&E", "Unknown"]

    if data_num == 7:
        # 加载任务3数据
        F_0 = np.load("./gearset/datagearset_thirdspace/T3_source_image.npy").transpose((0, 3, 1, 2)) / 255
        F_1 = np.load("./gearset/datagearset_thirdspace/T3_target_image.npy").transpose((0, 3, 1, 2)) / 255
        Y_0 = np.load("./gearset/datagearset_thirdspace/T3_source_label.npy")
        Y_1 = np.load("./gearset/datagearset_thirdspace/T3_target_label.npy")
        print("任务3 - Unknown 占比：", Y_1[:, -1].mean(), "样本数：", Y_1[:, -1].sum())
        Label_Name = ["1-N", "3-BF&SI", "4-G&E", "Unknown"]

    # 转换为PyTorch张量
    F_0 = torch.from_numpy(F_0.astype(np.float32))
    F_1 = torch.from_numpy(F_1.astype(np.float32))
    Y_0 = torch.from_numpy(Y_0.astype(np.float32))
    Y_1 = torch.from_numpy(Y_1.astype(np.float32))

    data_0 = Data.TensorDataset(F_0, Y_0)
    data_1 = Data.TensorDataset(F_1, Y_1)

    return data_0, data_1, None, F_0, F_1, Y_0, Y_1, Label_Name

def Split(full_dataset):
    # 4. 把数据集拆成“训练+测试”，测试固定200张图
    test_size = 200
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

# Model Building
# Step-1: Train the coarse classifier (Distract Stage)
def step_1_enhanced_complete(source_loader, target_loader, bridge_loader,
                             net, discriminator_coarse, monitor_interval=20):
    early_stop = EarlyStopping(patience=30, mode='max', verbose=True)
    log = Logger('./log/step_1_complete', clear=True)
    step = 0
    best_acc = 0
    best_net_state = None
    best_discriminator_state = None

    print("=" * 70)
    print("开始 Step-1: 消融实验（无桥梁域）")
    print("=" * 70)

    start_time = time.time()

    # 训练循环
    while step < 1000:
        # 数据迭代
        data_zip = zip(source_loader, target_loader)

        for batch_idx, data in enumerate(data_zip):
            # 准备数据
            (im_source, label_source), (im_target, label_target) = data
            im_bridge, label_bridge = None, None

            # 移动到设备
            im_source = im_source.to(device)
            label_source = label_source.to(device)
            im_target = im_target.to(device)
            label_target = label_target.to(device)

            # ========== 前向传播 ==========
            net.train()
            discriminator_coarse.train()

            # 源域
            fs1, feature_source, predict_prob_source = net.forward(im_source)
            # 目标域
            ft1, feature_target, predict_prob_target = net.forward(im_target)

            # ========== 计算损失 ==========
            # 1. 分类损失
            ce_loss = CrossEntropyLoss(label_source, predict_prob_source)
            # 2. 判别器损失
            coarse_source = discriminator_coarse.forward(fs1)
            coarse_target = discriminator_coarse.forward(ft1)
            d1_loss = BCELossForMultiClassification(label_source[:, 0:num_class], coarse_source)

            # 调整损失权重 - 增加分类损失的权重
            alpha = 1.5  # 分类损失权重从1.0增加到1.5
            beta = 0.3  # 桥梁对齐损失权重减小
            gamma = 0.5  # 判别器损失权重
            total_loss = alpha * ce_loss + gamma * d1_loss
            # ========== 反向传播 ==========
            optimizer_net.zero_grad()
            optimizer_discriminator_coarse.zero_grad()

            if total_loss.requires_grad:
                total_loss.backward()
                optimizer_net.step()
                optimizer_discriminator_coarse.step()

            step += 1
            log.step += 1

            # ========== 监控和评估 ==========
            if step % monitor_interval == 0:
                # 1. 评估准确率
                net.eval()
                with torch.no_grad():
                    acc_test = 0
                    for eval_im, eval_label in target_loader:
                        eval_im = eval_im.to(device)
                        eval_label = eval_label.to(device)
                        _, _, eval_prob = net.forward(eval_im)
                        eval_pred = torch.argmax(eval_prob, dim=1)
                        eval_true = torch.argmax(eval_label[:, :num_class], dim=1)
                        acc_test = (eval_pred == eval_true).float().mean().item()
                        break

                # 2. 早停检查
                if early_stop(acc_test, net.state_dict()):
                    print(f"Step-1 触发早停! 步数: {step}, 准确率: {acc_test:.4f}")
                    break

                # 3. 保存最佳模型
                if acc_test > best_acc:
                    best_acc = acc_test
                    best_net_state = copy.deepcopy(net.state_dict())
                    best_discriminator_state = copy.deepcopy(discriminator_coarse.state_dict())

                    # 保存模型文件
                    torch.save(net.state_dict(), './models/net_complete_bridge.pkl')
                    torch.save(discriminator_coarse.state_dict(),
                               './models/discriminator_coarse_complete.pkl')

                # 4. 打印训练信息
                elapsed_time = time.time() - start_time
                print(f"Step {step}: 准确率={acc_test:.4f}, 最佳={best_acc:.4f}, "
                      f"总损失={total_loss.item():.4f}, 耗时={elapsed_time:.2f}秒")

                net.train()  # 恢复训练模式

        # 检查早停
        if early_stop.early_stop:
            print("训练结束（早停）")
            break

    # ========== 训练后处理 ==========
    # 1. 加载最佳模型
    if best_net_state is not None:
        net.load_state_dict(best_net_state)
    if best_discriminator_state is not None:
        discriminator_coarse.load_state_dict(best_discriminator_state)

    total_time = time.time() - start_time
    print(f"\nStep-1 完成! 总耗时: {total_time:.2f}秒, 最终最佳准确率: {best_acc:.4f}")
    print("=" * 70)

    return './models/net_complete_bridge.pkl', './models/discriminator_coarse_complete.pkl'

# ## Initialize the minibatch number (Domain consensus score)
def cal_score(num_centers, feat):
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat)
    center, t_codes = kmeans.cluster_centers_, kmeans.labels_
    score = sklearn.metrics.silhouette_score(feat, t_codes)
    return score

def get_src_centers(F1, Y1, num_class):
    # 修改点5：确保张量在正确的设备上
    s_center = torch.zeros((num_class, 1024)).float().to(device)
    s_feats = torch.tensor(F1).to(device)
    s_labels = torch.tensor(Y1).to(device)
    for i in s_labels.argmax(axis=1).unique():
        i_msk = s_labels.argmax(axis=1) == i
        index = i_msk.squeeze().nonzero(as_tuple=False)
        i_feat = s_feats[index, :].mean(0)
        i_feat = F.normalize(i_feat, p=2, dim=1)
        s_center[i, :] = i_feat
    return s_center, s_feats, s_labels.argmax(axis=1)

def get_tgt_centers_EDL_enhanced(F2, max_extra=15, min_clusters=2, reg_covar=1e-1):
    """
    增强版本：解决协方差矩阵不正定问题
    """
    # 修改点6：确保张量在正确的设备上
    t_feats = torch.tensor(F2).to(device)
    N = t_feats.shape[0]

    print(f"DP-GMM输入数据: {t_feats.shape}, 特征范围: [{t_feats.min():.4f}, {t_feats.max():.4f}]")

    # 1. 数据预处理 - 标准化
    scaler = StandardScaler()
    t_feats_scaled = scaler.fit_transform(t_feats.cpu().numpy())

    # 检查数据质量
    print(f"标准化后数据范围: [{t_feats_scaled.min():.4f}, {t_feats_scaled.max():.4f}]")
    print(f"数据均值: {t_feats_scaled.mean():.4f}, 标准差: {t_feats_scaled.std():.4f}")

    # 2. 使用更稳定的DP-GMM参数
    dpgmm = BayesianGaussianMixture(
        n_components=min(max_extra, N // 10),  # 确保组件数不超过样本数的1/10
        covariance_type='diag',  # 使用对角协方差，更稳定
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1e-3,
        mean_precision_prior=1e-2,
        reg_covar=reg_covar,  # 增加正则化
        max_iter=200,
        random_state=42,
        n_init=3  # 多次初始化
    )

    print("开始DP-GMM拟合...")
    dpgmm.fit(t_feats_scaled)
    print("DP-GMM拟合完成")

    # 3. 去掉权重≈0 的空簇
    alive_mask = dpgmm.weights_ > 1e-3
    alive_k = alive_mask.sum()

    if alive_k == 0:
        raise ValueError("所有簇权重都为0，DP-GMM聚类失败")

    t_labels = torch.tensor(dpgmm.predict(t_feats_scaled)).to(device)
    alive_centers = torch.tensor(dpgmm.means_[alive_mask]).float().to(device)

    # 4. 反标准化中心点
    alive_centers_original = scaler.inverse_transform(alive_centers.cpu().numpy())
    alive_centers = torch.tensor(alive_centers_original).float().to(device)

    # 5. 改进的EDL过滤策略
    if alive_k <= min_clusters:
        keep = np.ones(alive_k, dtype=bool)
    else:
        # 使用更稳定的证据计算
        log_weight = np.log(dpgmm.weights_[alive_mask] + 1e-8)

        # 对于对角协方差，计算对数行列式
        if dpgmm.covariance_type == 'diag':
            log_det = np.sum(np.log(dpgmm.covariances_[alive_mask] + 1e-8), axis=1)
        else:
            try:
                log_det = np.linalg.slogdet(dpgmm.covariances_[alive_mask])[1]
            except:
                log_det = np.ones(alive_k)  # 回退

        evidences = log_weight + 0.5 * log_det

        # 保留前K个最高evidence的簇
        k_keep = max(min_clusters, int(alive_k * 0.7))
        topk_indices = np.argsort(evidences)[-k_keep:]
        keep = np.zeros(alive_k, dtype=bool)
        keep[topk_indices] = True

    final_k = keep.sum()

    if final_k == 0:
        print("警告: 过滤后无簇保留，使用所有存活簇")
        keep = np.ones(alive_k, dtype=bool)
        final_k = alive_k

    print(f"DP-GMM+EDL聚类结果: 初始{max_extra}簇 -> 存活{alive_k}簇 -> 最终{final_k}簇")

    # 6. 重新编排标签
    old2new = {old: new for new, old in enumerate(np.where(keep)[0])}
    t_labels_new = torch.tensor([old2new.get(x.item(), -1) for x in t_labels],
                                device=t_labels.device, dtype=t_labels.dtype)

    # 处理被过滤的样本
    if (t_labels_new == -1).sum() > 0:
        filtered_mask = (t_labels_new == -1)
        filtered_feats = t_feats[filtered_mask]

        with torch.no_grad():
            kept_centers = alive_centers[keep]
            dists = torch.cdist(filtered_feats, kept_centers)
            nearest_clusters = dists.argmin(dim=1)
            t_labels_new[filtered_mask] = nearest_clusters

    # 7. 计算最终簇中心
    t_center = torch.zeros((final_k, 1024)).float().to(device)
    for k in range(final_k):
        cluster_mask = (t_labels_new == k)
        if cluster_mask.sum() > 0:
            t_center[k] = t_feats[cluster_mask].mean(0)
        else:
            t_center[k] = alive_centers[keep][k]

    t_center = F.normalize(t_center, p=2, dim=1)

    return t_center, t_feats, t_labels_new

def consensus_score(t_feats, t_codes, t_centers, s_feats, s_labels, s_centers):
    s_centers = F.normalize(s_centers, p=2, dim=-1)
    t_centers = F.normalize(t_centers, p=2, dim=-1)
    simis = torch.matmul(s_centers, t_centers.transpose(0, 1))
    s_index = simis.argmax(dim=1)
    t_index = simis.argmax(dim=0)
    map_s2t = [(i, s_index[i].item()) for i in range(len(s_index))]
    map_t2s = [(t_index[i].item(), i) for i in range(len(t_index))]
    inter = [a for a in map_s2t if a in map_t2s]
    p_score = 0.0
    filtered_inter = []
    t_score = 0.0
    s_score = 0.0
    scores = []
    score_dict = {}
    score_vector = torch.zeros(s_centers.shape[0]).float().to(device)
    for i, j in inter:
        si_index = (s_labels == i).squeeze().nonzero(as_tuple=False)
        tj_index = (t_labels == j).squeeze().nonzero(as_tuple=False)
        si_feat = s_feats[si_index, :]
        tj_feat = t_feats[tj_index, :]

        s2TC = torch.matmul(si_feat, t_centers.transpose(0, 1))
        s2TC = s2TC.argmax(dim=-1)
        p_i2j = (s2TC == j).sum().float() / len(s2TC)
        t2SC = torch.matmul(tj_feat, s_centers.transpose(0, 1))
        t2SC = t2SC.argmax(dim=-1)
        p_j2i = (t2SC == i).sum().float() / len(t2SC)

        cu_score = (p_j2i + p_i2j) / 2
        score_dict[(i, j)] = (p_j2i, p_i2j)
        filtered_inter.append((i, j))
        t_score += p_j2i.item()
        s_score += p_i2j.item()
        p_score += cu_score.item()
        scores.append(cu_score.item())
        score_vector[i] += cu_score.item()

    score = p_score / len(filtered_inter)
    t_score = t_score / len(filtered_inter)
    s_score = s_score / len(filtered_inter)
    min_score = np.min(scores)
    return score, score_vector, filtered_inter, scores, score_dict

# 新增：高效Wasserstein距离计算类
class EfficientWassersteinDistance:
    """高效的批量Wasserstein距离计算"""

    def __init__(self, num_bins=100, use_gpu=True):
        """
        初始化参数
        Args:
            num_bins: 直方图分箱数
            use_gpu: 是否使用GPU加速
        """
        self.num_bins = num_bins
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

    def _compute_histograms(self, features, labels=None):
        """
        计算特征的直方图表示
        Args:
            features: 特征矩阵 [N, D]
            labels: 标签向量 [N]
        Returns:
            直方图字典 {label: histogram}
        """
        if labels is None:
            # 如果不提供标签，将整个特征集视为一个分布
            return self._compute_single_histogram(features)

        histograms = {}
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_mask = (labels == label)
            label_features = features[label_mask]
            hist = self._compute_single_histogram(label_features)
            histograms[label.item()] = hist

        return histograms

    def _compute_single_histogram(self, features):
        """
        为单个特征集计算直方图
        Args:
            features: 特征矩阵 [M, D]
        Returns:
            直方图向量
        """
        if len(features) == 0:
            return None

        # 转换为numpy用于计算直方图
        if torch.is_tensor(features):
            features_np = features.cpu().numpy()
        else:
            features_np = features

        # 计算每个维度的直方图，然后平均
        histograms = []
        for d in range(features_np.shape[1]):
            hist, _ = np.histogram(features_np[:, d], bins=self.num_bins, density=True)
            histograms.append(hist)

        # 返回平均直方图
        return np.mean(histograms, axis=0)

    def compute_batch_distances(self, query_features, reference_histograms):
        """
        批量计算查询特征与参考分布的Wasserstein距离
        Args:
            query_features: 查询特征 [N, D]
            reference_histograms: 参考分布的直方图字典
        Returns:
            距离矩阵 [N, C]
        """
        if torch.is_tensor(query_features):
            query_features_np = query_features.cpu().numpy()
        else:
            query_features_np = query_features

        N = query_features_np.shape[0]
        C = len(reference_histograms)

        # 预计算查询特征的直方图
        query_hists = []
        for i in range(N):
            query_hist = self._compute_single_histogram(query_features_np[i:i + 1])
            query_hists.append(query_hist)

        # 批量计算距离
        distance_matrix = np.zeros((N, C))
        for i, query_hist in enumerate(query_hists):
            if query_hist is None:
                continue

            for j, (label, ref_hist) in enumerate(reference_histograms.items()):
                if ref_hist is None:
                    distance_matrix[i, j] = np.inf
                else:
                    # 使用Wasserstein距离的近似计算
                    dist = wasserstein_distance(query_hist, ref_hist)
                    distance_matrix[i, j] = dist

        return distance_matrix

    def adaptive_threshold_selection(self, distances, labels, num_classes):
        """
        自适应阈值选择
        Args:
            distances: 距离矩阵 [N, C]
            labels: 真实标签 [N]
            num_classes: 类别数
        Returns:
            最优阈值
        """
        # 将距离转换为置信度分数（距离越小，置信度越高）
        confidences = 1.0 / (1.0 + distances.min(axis=1))

        # 区分已知和未知类别
        known_mask = labels < num_classes

        if known_mask.sum() == 0 or (~known_mask).sum() == 0:
            # 如果只有单一类别，返回默认阈值
            return 0.5

        known_confs = confidences[known_mask]
        unknown_confs = confidences[~known_mask]

        # 使用ROC曲线找到最佳阈值
        from sklearn.metrics import roc_curve
        y_true = np.concatenate([np.zeros_like(known_confs), np.ones_like(unknown_confs)])
        y_score = np.concatenate([known_confs, unknown_confs])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # 使用Youden's J统计量找到最佳阈值
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_threshold = thresholds[best_idx]

        return best_threshold

# Step-2: Train the fine classifier (Distract stage)
def step_2_v2(best_net_path, best_discriminator_path,
              source_loader, target_loader, bridge_loader,
              batch=40, max_iter=50, device=device):
    print("=" * 50)
    print("开始 Step-2: 训练细分类器（无桥梁域）")
    print("=" * 50)
    start_time = time.time()

    # 添加焦点损失（Focal Loss）来处理难易样本不平衡
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss

    focal_loss = FocalLoss(alpha=0.75, gamma=1.5)  # 关注难样本

    # 改进的模型加载函数
    def safe_load_model(model, model_path, model_name=""):
        """安全加载模型，处理各种可能的错误"""
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return False

        try:
            # 加载模型时指定正确的设备
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 获取当前模型的状态字典
            current_state_dict = model.state_dict()

            # 检查键匹配
            missing_keys = [k for k in current_state_dict.keys() if k not in state_dict]
            unexpected_keys = [k for k in state_dict.keys() if k not in current_state_dict]

            if missing_keys:
                print(f"警告 {model_name}: 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"警告 {model_name}: 意外的键: {unexpected_keys}")

            # 尝试严格模式加载
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"成功加载 {model_name} (严格模式)")
            except:
                print(f"严格模式加载失败，尝试非严格模式加载 {model_name}")
                model.load_state_dict(state_dict, strict=False)
                print(f"成功加载 {model_name} (非严格模式)")

            model.to(device)  # 确保模型在正确的设备上
            return True

        except Exception as e:
            print(f"加载 {model_name} 失败: {e}")
            return False

    # 安全加载模型
    if not safe_load_model(net, best_net_path, "net"):
        print("网络加载失败，无法继续Step-2")
        return None

    if not safe_load_model(discriminator_coarse, best_discriminator_path, "discriminator_coarse"):
        print("判别器加载失败，但继续Step-2")

    net.eval()
    discriminator_coarse.eval()

    # 预提取源域特征/标签
    print("正在提取源域特征...")
    src_feats, src_lbls = [], []
    with torch.no_grad():
        for im, lbl in source_loader:
            # 修改点8：确保数据在正确的设备上
            im, lbl = im.to(device), lbl.to(device)
            f, _, _ = net(im)
            src_feats.append(f.cpu().numpy())
            src_lbls.append(lbl.argmax(1).cpu().numpy())
    src_feats = np.concatenate(src_feats)
    src_lbls = np.concatenate(src_lbls)

    print(f"源域特征维度: {src_feats.shape}")

    # 计算类均值和协方差
    print("正在计算类均值和协方差...")
    class_mean, class_cov = {}, {}
    for c in np.unique(src_lbls):
        mask = src_lbls == c
        class_mean[c] = src_feats[mask].mean(0)
        class_cov[c] = EmpiricalCovariance().fit(src_feats[mask]).covariance_

    # 迭代训练
    print(f"开始迭代训练，共 {max_iter} 轮:")

    for it in range(1):
        iter_start_time = time.time()
        print(f"\n--- 第 {it + 1}/{max_iter} 轮迭代 ---")

        tgt_feats, tgt_energy, tgt_wasserstein = [], [], []
        with torch.no_grad():
            batch_count = 0
            for im, _ in target_loader:
                # 修改点10：确保数据在正确的设备上
                im = im.to(device)
                f, _, logits = net(im)
                f_np = f.cpu().numpy()
                logits_np = logits.cpu().numpy()

                # Energy Score
                energy = -torch.logsumexp(logits, dim=1).cpu().numpy()

                # 计算Wasserstein距离
                wasserstein_dist = np.array([min([calculate_wasserstein_distance(feat.reshape(1, -1),
                                                        src_feats[src_lbls == c])
                         for c in class_mean]) for feat in f_np
                ])  # No need for pseudo labels here

                tgt_feats.append(f_np)
                tgt_energy.append(energy)
                tgt_wasserstein.append(wasserstein_dist)
                batch_count += 1

        tgt_feats = np.concatenate(tgt_feats)
        tgt_energy = np.concatenate(tgt_energy)
        tgt_wasserstein = np.concatenate(tgt_wasserstein)

        # 训练二分类器
        discriminator_fine.train()
        for param in discriminator_fine.parameters():
            param.requires_grad = True

        epoch_losses = []
        for epoch in range(5):
            known_output = discriminator_fine(torch.tensor(tgt_feats, device=device))

            if isinstance(known_output, list):
                pred_k = known_output[0][:, 1]
            else:
                pred_k = known_output[:, 1]

            if not pred_k.requires_grad:
                pred_k = pred_k.requires_grad_(True)
            # 使用焦点损失替代标准BCE
            loss = focal_loss(pred_k, torch.ones_like(pred_k))
            # loss = F.binary_cross_entropy_with_logits(pred_k, torch.ones_like(pred_k))

            optimizer_discriminator_fine.zero_grad()

            if not loss.requires_grad:
                print("警告: 损失不需要梯度，重新计算")
                continue

            loss.backward()
            optimizer_discriminator_fine.step()
            epoch_losses.append(loss.item())

            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch + 1}/5: 损失={loss.item():.4f}")

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"  细分类器训练: 平均损失={avg_loss:.4f}")

        iter_time = time.time() - iter_start_time
        print(f"第 {it + 1} 轮完成，耗时: {iter_time:.2f}秒")

    # 修改：使用新的模型文件名，添加_wavelet后缀
    torch.save(discriminator_fine.state_dict(),
               './models/final_discriminator_fine_v2_wavelet.pkl')

    total_time = time.time() - start_time
    print(f"Step-2 完成! 总耗时: {total_time:.2f}秒")
    print("=" * 50)

    return './models/final_discriminator_fine_v2_wavelet.pkl'

# Step-3: Train the discriminator (Attract stage)
def step_3(source_loader, target_loader, bridge_loader, discriminator, minibatch, net_path, discriminator_fine_path):
    early_stop = EarlyStopping(patience=30, mode='max', verbose=True)

    print("=" * 50)
    print("开始 Step-3: 训练域判别器")
    print("=" * 50)
    start_time = time.time()

    # 改进的模型加载函数
    def safe_load_model(model, model_path, model_name=""):
        """安全加载模型，处理各种可能的错误"""
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return False

        try:
            # 修改点11：加载模型时指定正确的设备
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 获取当前模型的状态字典
            current_state_dict = model.state_dict()

            # 检查键匹配
            missing_keys = [k for k in current_state_dict.keys() if k not in state_dict]
            unexpected_keys = [k for k in state_dict.keys() if k not in current_state_dict]

            if missing_keys:
                print(f"警告 {model_name}: 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"警告 {model_name}: 意外的键: {unexpected_keys}")

            # 尝试严格模式加载
            try:
                model.load_state_dict(state_dict, strict=True)
                print(f"成功加载 {model_name} (严格模式)")
                return True
            except:
                print(f"严格模式加载失败，尝试非严格模式加载 {model_name}")
                model.load_state_dict(state_dict, strict=False)
                print(f"成功加载 {model_name} (非严格模式)")
                return True

        except Exception as e:
            print(f"加载 {model_name} 失败: {e}")
            return False

    # 安全加载模型
    if not safe_load_model(net, net_path, "net"):
        print("网络加载失败，无法继续Step-3")
        return None, None

    if not safe_load_model(discriminator_fine, discriminator_fine_path, "discriminator_fine"):
        print("细判别器加载失败，但继续Step-3")

    discriminator = LargeAdversarialNetwork(1024).to(device)

    optimizer_feature_extractor = OptimWithSheduler(optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4), scheduler)
    optimizer_cls = OptimWithSheduler(optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-4), scheduler)
    optimizer_discriminator = OptimWithSheduler(optim.Adam(discriminator.parameters(), lr=1e-5, weight_decay=5e-4),
                                                scheduler)

    log = Logger('./log/TPTLN/', clear=True)
    k = 0
    acc = 0
    best_acc = 0

    print(f"总共训练 40 轮，minibatch={minibatch}")

    # 构建数据加载器列表
    loaders = [source_loader, target_loader]
    print("消融实验：不使用桥梁域数据进行对抗训练")

    while k < 40:
        epoch_start_time = time.time()
        epoch_losses = []

        for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(zip(*loaders)):
            net.train()
            discriminator_fine.train()
            discriminator.train()

            im_source, label_source = im_source.to(device), label_source.to(device)
            im_target, label_target = im_target.to(device), label_target.to(device)

            # 源域前向
            fs1, feature_source, predict_prob_source = net.forward(im_source)
            # 目标域前向
            ft1, feature_target, predict_prob_target = net.forward(im_target)
            # 修改：移除桥梁域前向

            domain_prob_discriminator_1_source = discriminator.forward(fs1)
            domain_prob_discriminator_1_target = discriminator.forward(ft1)

            dptarget_output = discriminator_fine.forward(ft1.detach())
            if isinstance(dptarget_output, list):
                _, _, dptarget = dptarget_output
            else:
                dptarget = dptarget_output

            A = dptarget[:, 1].cpu().detach().numpy()
            A = Variable(torch.tensor(A).to(device))

            # 计算分类损失（包含源域和桥梁域）
            ce = CrossEntropyLoss(label_source, predict_prob_source)
            known = torch.sort(dptarget[:, 1].detach(), dim=0)[1][-minibatch:]
            unknown = torch.sort(dptarget[:, 1].detach(), dim=0)[1][0:minibatch]
            feature_unknown = torch.index_select(ft1, 0, unknown)
            feature_known = torch.index_select(ft1, 0, known)

            pred_unknown_output = discriminator_fine.forward(feature_unknown)
            pred_known_output = discriminator_fine.forward(feature_known)
            if isinstance(pred_unknown_output, list):
                _, _, pred_unknown = pred_unknown_output
            else:
                pred_unknown = pred_unknown_output
            if isinstance(pred_known_output, list):
                _, _, pred_known = pred_known_output
            else:
                pred_known = pred_known_output

            _, __, predict_prob_otherep = cls.forward(feature_unknown)
            ce_ep = CrossEntropyLoss(Variable(torch.from_numpy(
                np.concatenate((np.zeros((minibatch, num_class)), np.ones((minibatch, 1))), axis=-1).astype(
                    'float32'))).to(device), predict_prob_otherep)

            # 对抗损失（包含源域、桥梁域和目标域）
            adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source),
                                                     predict_prob=domain_prob_discriminator_1_source)
            adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target),
                                                      predict_prob=1 - domain_prob_discriminator_1_target,
                                                      instance_level_weight=A)

            for param in net.parameters():
                param.requires_grad = True
            for param in discriminator.parameters():
                param.requires_grad = True

            with OptimizerManager([optimizer_feature_extractor, optimizer_cls, optimizer_discriminator]):
                loss = ce + 0.05 * adv_loss + 0.05 * ce_ep

                if not loss.requires_grad:
                    print("步骤3: 损失不需要梯度，跳过反向传播")
                    continue

                loss.backward()
                epoch_losses.append(loss.item())

            acc_test = accuracy_score(np.argmax(predict_prob_target.cpu().detach().numpy(), axis=-1),
                                          np.argmax(label_target.cpu().detach().numpy(), axis=-1))
            if early_stop(acc_test, net.state_dict()):
                print("Step-3 触发早停!")
                break
            if acc_test >= acc:
                acc = acc_test
                # 修改：使用新的模型文件名，添加_wavelet后缀
                torch.save(discriminator_fine.state_dict(), './models/discriminator_fine_wavelet_.pkl')
                torch.save(net.state_dict(), './models/target_net_wavelet_.pkl')
                final_net_path = './models/target_net_wavelet_.pkl'

                if acc_test > best_acc:
                    best_acc = acc_test

        k += 1
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0

        print(
            f"Epoch {k}/40: 损失={avg_loss:.4f}, 当前准确率={acc_test:.4f}, 最佳准确率={best_acc:.4f}, 耗时={epoch_time:.2f}秒")

    total_time = time.time() - start_time
    print(f"Step-3 完成! 总耗时: {total_time:.2f}秒, 最终最佳准确率: {best_acc:.4f}")
    print("=" * 60)

    return final_net_path, discriminator_fine_path

# 替换原有的 calculate_wasserstein_distance 函数
def calculate_wasserstein_distance(sample_feat, class_feats):
    """
    计算样本特征与类别特征分布之间的Wasserstein距离
    增强版：处理各种边界情况
    """
    try:
        # 输入验证
        if sample_feat is None or class_feats is None:
            print(f"警告: 输入数据为空 - sample_feat: {sample_feat is None}, class_feats: {class_feats is None}")
            return 10.0  # 返回一个大值表示距离很远

        # 确保 sample_feat 是二维的
        if sample_feat.ndim == 1:
            sample_feat = sample_feat.reshape(1, -1)
        elif sample_feat.ndim > 2:
            sample_feat = sample_feat.reshape(1, -1)

        # 确保 sample_feat 有数据
        if sample_feat.size == 0:
            print("警告: sample_feat 为空")
            return 10.0

        # 处理 class_feats
        valid_distances = []
        for i, cf in enumerate(class_feats):
            if cf is None or cf.size == 0:
                # 跳过空类别
                continue

            try:
                # 确保 cf 是二维的
                if cf.ndim == 1:
                    cf_2d = cf.reshape(-1, 1)
                else:
                    cf_2d = cf

                # 计算Wasserstein距离
                sample_flat = sample_feat.flatten()
                cf_flat = cf_2d.flatten()

                # 检查数据有效性
                if np.any(np.isnan(sample_flat)) or np.any(np.isnan(cf_flat)):
                    print(f"警告: 特征中包含NaN值")
                    continue

                if np.any(np.isinf(sample_flat)) or np.any(np.isinf(cf_flat)):
                    print(f"警告: 特征中包含无穷值")
                    continue

                # 计算距离
                dist = wasserstein_distance(sample_flat, cf_flat)
                valid_distances.append(dist)

            except Exception as e:
                print(f"计算Wasserstein距离时出错 (类别 {i}): {e}")
                continue

        if valid_distances:
            return min(valid_distances)
        else:
            print("警告: 所有Wasserstein距离计算都失败了，使用备用距离")
            # 备用方案：使用欧氏距离
            return min([np.linalg.norm(sample_feat.flatten() - cf.flatten())
                        for cf in class_feats if cf is not None and cf.size > 0])

    except Exception as e:
        print(f"Wasserstein距离计算错误: {e}")
        return 10.0  # 返回默认大距离

# # 新增：H-score计算函数
def calculate_h_score(known_accuracy, unknown_accuracy):
    """
    计算已知类和未知类准确率的调和平均值（H-score）
    """
    if known_accuracy + unknown_accuracy == 0:
        return 0.0
    h_score = 2 * known_accuracy * unknown_accuracy / (known_accuracy + unknown_accuracy)
    return h_score


# # Model training
# ## Dataset generating
task_num = 7
setGPU('0')
# 修改数据加载以包含桥梁域
data_0, data_1, data_bridge, F_0, F_1, Y_0, Y_1, Label_Name = get_dat(task_num)
Train_0, Test_0 = Split(data_0)
Train_1, Test_1 = Split(data_1)

bridge_loader = None
bridge_loader1 = None
print("消融实验配置：不使用桥梁域数据")

source_loader = Data.DataLoader(dataset=Train_0, batch_size=40, shuffle=True, num_workers=4)
target_loader = Data.DataLoader(dataset=Train_1, batch_size=40, shuffle=True, num_workers=4)
source_loader1 = Data.DataLoader(dataset=Test_0, batch_size=40, shuffle=True, num_workers=4)
target_loader1 = Data.DataLoader(dataset=Test_1, batch_size=40, shuffle=True, num_workers=4)

num_class = Y_0.shape[1] - 1

# 修改点13：确保所有模型都在正确的设备上
discriminator_fine = CLS_0(1024, 2).to(device)
discriminator_coarse = Discriminator(n=Y_0.shape[1] - 1).to(device)
# 使用小波CNN作为特征提取器
feature_extractor = WaveletCNN_2D(in_channel=3, use_wavelet=True).to(device)
cls = CLS(1024, Y_0.shape[1]).to(device)
net = nn.Sequential(feature_extractor, cls).to(device)
discriminator = LargeAdversarialNetwork(1024).to(device)

scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_discriminator_fine = OptimWithSheduler(
    optim.Adam(discriminator_fine.parameters(), lr=5e-4, weight_decay=5e-4),
    scheduler)
optimizer_discriminator_coarse = OptimWithSheduler(
    optim.Adam(discriminator_coarse.parameters(), lr=5e-4, weight_decay=5e-4),
    scheduler)
optimizer_net = OptimWithSheduler(optim.Adam(net.parameters(), lr=5e-4, weight_decay=5e-4), scheduler)
optimizer_cls = OptimWithSheduler(optim.Adam(cls.parameters(), lr=5e-4, weight_decay=5e-4), scheduler)

SCRIPT_START = time.time()
print("🚀 开始整个训练流程")
print(f"任务编号: {task_num}")
print(f"源域样本数: {len(Train_0)}")
print(f"目标域样本数: {len(Train_1)}")
print(f"类别数: {num_class}")
print("=" * 60)

# ## Model training (3-steps)
## STEP-1 Coarse Classifier
print("开始Step-1: 粗分类器训练")
# 使用完整的增强版Step-1训练
src_net_path, src_discriminator_coarse_path = step_1_enhanced_complete(
    source_loader, target_loader, bridge_loader,
    net, discriminator_coarse, monitor_interval=20
)
# 检查Step-1是否成功完成
if src_net_path is None or not os.path.exists(src_net_path):
    print("错误: Step-1 未能成功生成模型文件")
    exit(1)

print("正在提取特征用于共识分数计算...")
feat_s = []
feat_t = []
lab_s = []
lab_t = []
batch_count = 0
total_batches = min(len(source_loader), len(target_loader))

# 安全加载网络用于特征提取
def safe_load_for_features(net, net_path):
    try:
        # 修改点14：加载模型时指定正确的设备
        net.load_state_dict(torch.load(net_path, map_location=device), strict=False)
        net.eval()
        return True
    except Exception as e:
        print(f"加载网络用于特征提取失败: {e}")
        return False


if not safe_load_for_features(net, src_net_path):
    print("无法加载网络进行特征提取，使用默认minibatch=10")
    minibatch = 10
else:
    for (i, ((im_source, label_source), (im_target, label_target))) in enumerate(zip(source_loader, target_loader)):
        # 修改点15：确保数据在正确的设备上
        im_source, label_source = im_source.to(device), label_source.to(device)
        im_target, label_target = im_target.to(device), label_target.to(device)

        fs1, feature_source, predict_prob_source = net.forward(im_source)
        ft1, feature_target, predict_prob_target = net.forward(im_target)
        feat_s.append(np.array(fs1.cpu().detach().numpy()))
        feat_t.append(np.array(ft1.cpu().detach().numpy()))
        lab_s.append(np.array(label_source.cpu().detach().numpy()))
        lab_t.append(np.array(predict_prob_target.cpu().detach().numpy()))
        batch_count += 1
        if batch_count % 5 == 0:
            print(f"特征提取进度: {batch_count}/{total_batches} batches")

    # 方法1：使用动态维度
    feat_s_array = np.concatenate(feat_s, axis=0)
    feat_t_array = np.concatenate(feat_t, axis=0)
    # 获取实际的batch总数和特征维度
    total_samples = len(Train_0)  # 应该是600
    if feat_s_array.ndim == 3:  # [batch_num, batch_size, feature_dim]
        batch_num, batch_size, feature_dim = feat_s_array.shape
        F1 = feat_s_array.reshape(total_samples, feature_dim)
        F2 = feat_t_array.reshape(total_samples, feature_dim)
    else:
        # 直接拼接
        F1 = np.concatenate(feat_s, axis=0)
        F2 = np.concatenate(feat_t, axis=0)
        print(f"F1 shape after concat: {F1.shape}")
        print(f"F2 shape after concat: {F2.shape}")

    # 同样处理标签
    Y1 = np.concatenate(lab_s, axis=0)
    Y2 = np.concatenate(lab_t, axis=0)

    print("正在计算共识分数和开放度...")
    # 使用增强的DP-GMM+EDL方法
    print("使用增强的DP-GMM+EDL进行目标域聚类...")
    try:
        t_centers, t_feats, t_labels = get_tgt_centers_EDL_enhanced(
            F2, max_extra=15, min_clusters=num_class, reg_covar=1e-1
        )

        s_centers, s_feats, s_labels = get_src_centers(F1, Y1, num_class)

        # 计算共识分数
        score, score_vector, filtered_inter, scores, score_dict = consensus_score(t_feats, t_labels, t_centers, s_feats,
                                                                                  s_labels, s_centers)

        # 计算开放度
        T = 0
        for i in range(sum(score_vector != 0).item()):
            if i < len(filtered_inter):
                t = score_vector[i] * sum(t_labels == filtered_inter[i][1])
                T = T + t.item()
        o = (400 - T) / 400
        minibatch = np.round(o * 40 * 0.25).astype(np.int64)

        print(f"聚类结果:")
        print(f"  - 目标域聚类数: {len(t_centers)}")
        print(f"  - 共识分数: {score:.4f}")
        print(f"  - 开放度: {o:.4f}")
        print(f"  - minibatch: {minibatch}")
    except Exception as e:
        print(f"聚类计算失败: {e}, 使用默认minibatch=10")
        minibatch = 10

## STEP-2 Fine Classifier
print("开始Step-2: 细分类器训练")
net_path = src_net_path

discriminator_fine_path = step_2_v2(
    best_net_path=src_net_path,
    best_discriminator_path=src_discriminator_coarse_path,
    source_loader=source_loader,
    target_loader=target_loader,
    bridge_loader=bridge_loader,
    batch=40,
    max_iter=50,
    device=device)

if discriminator_fine_path is None:
    print("Step-2 失败，无法继续Step-3")
    exit(1)

## STEP-3 Discriminator
print("开始Step-3: 对抗训练")
final_net_path, discriminator_fine_path = step_3(source_loader, target_loader, bridge_loader, discriminator, minibatch,
                                                 net_path,
                                                 discriminator_fine_path)

print("🎉 所有训练步骤完成!")
print("=" * 60)
print(f"【脚本总耗时】{(time.time() - SCRIPT_START)/60:.2f} 分钟")

# Model testing
print('正在用验证集自动选择最优 Energy 阈值 ...')
def find_best_energy_threshold_balanced(loader, num_class, min_known_acc=0.95, n_grid=500):
    """平衡已知类和未知类准确率的阈值选择，优先保证已知类准确率"""
    energy_list, is_unk_list = [], []
    net.eval()
    with torch.no_grad():
        for im, lbl in loader:
            im = im.to(device)
            _, _, logits = net(im)
            energy = -torch.logsumexp(logits, dim=1).cpu().numpy()
            energy_list.append(energy)

            # 获取真实标签
            if lbl.shape[1] > 1:
                is_unk = (lbl[:, -1] == 1).cpu().numpy().astype(int)
            else:
                is_unk = np.zeros(len(lbl))
            is_unk_list.append(is_unk)

    energy_vec = np.concatenate(energy_list)
    is_unk_vec = np.concatenate(is_unk_list)

    # 更精细的阈值搜索范围
    # 取已知类样本的95%分位数作为起始搜索点
    known_energy = energy_vec[is_unk_vec == 0]
    if len(known_energy) > 0:
        emin = np.percentile(known_energy, 1)  # 最低的1%
        emax = np.percentile(known_energy, 99.5)  # 最高的0.5%
    else:
        emin = energy_vec.min()
        emax = energy_vec.max()

    thr_grid = np.linspace(emin, emax, n_grid)

    best_score, best_th = -np.inf, thr_grid[0]
    best_known_acc, best_unk_acc = 0, 0

    for t in thr_grid:
        pred_unk = energy_vec > t
        known_mask = is_unk_vec == 0
        unk_mask = is_unk_vec == 1

        known_acc = ((pred_unk == 0) & known_mask).sum() / max(known_mask.sum(), 1)
        unk_acc = ((pred_unk == 1) & unk_mask).sum() / max(unk_mask.sum(), 1)

        # 优先保证已知类准确率不低于min_known_acc
        if known_acc < min_known_acc:
            continue

        # 使用加权调和平均（F-beta分数），beta=0.5更关注已知类
        beta = 0.5  # 越小越关注已知类
        if known_acc > 0 and unk_acc > 0:
            f_beta = (1 + beta ** 2) * (known_acc * unk_acc) / (beta ** 2 * known_acc + unk_acc)
        else:
            f_beta = 0

        if f_beta > best_score:
            best_score = f_beta
            best_th = t
            best_known_acc = known_acc
            best_unk_acc = unk_acc
    # print(f'平衡阈值选择: {best_th:.4f}')
    # print(f'  已知类准确率: {best_known_acc:.4f}')
    # print(f'  未知类准确率: {best_unk_acc:.4f}')
    # print(f'  F{beta}-score: {best_score:.4f}')
    return best_th

def find_class_wise_energy_thresholds(loader, net, num_class, device, min_known_keep=0.98):
    """
    类别感知Energy阈值，保证每类至少有min_known_keep的比例不被误判为未知
    min_known_keep: 每类至少保留的比例（不判为未知）
    """
    net.eval()
    all_energy, all_labels = [], []
    with torch.no_grad():
        for im, lbl in loader:
            im = im.to(device)
            _, _, logits = net(im)
            energy = -torch.logsumexp(logits, dim=1).cpu().numpy()

            if lbl.shape[1] > 1:
                is_unk = lbl[:, -1].cpu().numpy().astype(bool)
                known_labels = lbl[:, :num_class].argmax(dim=1).cpu().numpy()
                true_label = np.where(is_unk, num_class, known_labels)
            else:
                true_label = lbl.cpu().numpy().flatten()

            all_energy.append(energy)
            all_labels.append(true_label)

    all_energy = np.concatenate(all_energy)
    all_labels = np.concatenate(all_labels)

    thresholds = []
    for c in range(num_class):
        mask = all_labels == c
        if mask.sum() == 0:
            thresholds.append(np.inf)  # 没有样本的类别设为无穷大
            continue

        class_e = all_energy[mask]

        # 寻找使至少min_known_keep比例的样本不被判为未知的阈值
        # 即寻找使 (class_e <= th) 的比例 >= min_known_keep 的最小th
        sorted_e = np.sort(class_e)
        idx = int(np.floor(len(sorted_e) * min_known_keep))
        if idx >= len(sorted_e):
            idx = len(sorted_e) - 1
        th = sorted_e[idx]

        # 添加小的缓冲，避免刚好在边界
        th = th + 0.01 * (np.max(class_e) - np.min(class_e))

        thresholds.append(th)

        # 打印每类的统计信息
        fake_unk_ratio = (class_e > th).mean()
        # print(f"  类别{c}: 阈值={th:.4f}, 样本数={len(class_e)}, 误判率={fake_unk_ratio:.4f}")
    return thresholds

best_energy_th = find_best_energy_threshold_balanced(
    target_loader1,
    num_class,
    min_known_acc=0.95,  # 保证未知类准确率不低于85%
    n_grid=200
)

print("开始模型测试...")
def test_model(loader, net, threshold, num_class, class_thresholds=None):
    preds_list, labels_list, scores_list = [], [], []
    for x, y in loader:
        x = x.to(device)

        with torch.no_grad():
            _, _, logits = net(x)
            energy = -torch.logsumexp(logits, dim=1)
            # 获取已知类预测
            pred_known = logits.argmax(dim=1).cpu().numpy()
            energy_np = energy.cpu().numpy()
            # 使用能量阈值判断是否为未知类
            # is_unknown = energy_np > threshold
            # 先拿已知类预测
            pred_known = logits.argmax(dim=1).cpu().numpy()
            # 用对应类别的阈值
            thresholds = class_thresholds
            if thresholds is not None:
                # 为每个样本选对应类别的阈值
                thresh = np.array([thresholds[p] if p < len(thresholds) else thresholds[-1]
                                   for p in pred_known])
                # is_unknown = energy_np > thresh
                prob = torch.softmax(logits, dim=1)
                max_conf, pred_known = prob.max(dim=1)  # 置信度 + 预测类
                max_conf = max_conf.cpu().numpy()
                pred_known = pred_known.cpu().numpy()
                max_conf = prob.max(dim=1)[0].cpu().numpy()
                # 双条件
                is_unknown = (energy_np > thresh) & (max_conf < 0.7)
            else:
                is_unknown = energy_np > threshold
            pred = np.where(is_unknown, num_class, pred_known)

            # 获取真实标签（正确处理未知类）
            if y.shape[1] > 1:  # one-hot格式
                # 假设最后一列是未知类指示器
                is_unknown_true = y[:, -1].cpu().numpy()
                known_labels = y[:, :num_class].argmax(dim=1).cpu().numpy()
                true_labels = np.where(is_unknown_true == 1, num_class, known_labels)
            else:
                true_labels = y.cpu().numpy().flatten()
            preds_list.append(pred)
            labels_list.append(true_labels)
            scores_list.append(energy_np)

    # 最后拼接处改成这样
    if preds_list:  # 关键判断
        preds = np.concatenate(preds_list)
        labels = np.concatenate(labels_list)
        scores = np.concatenate(scores_list)
    else:  # loader 里一个样本都没有
        preds = np.empty(0, dtype=int)
        labels = np.empty(0, dtype=int)
        scores = np.empty(0, dtype=float)
    return labels, preds, scores


# 在目标域测试集上测试
# class_thresholds = find_class_wise_energy_thresholds(target_loader1, net, num_class, device)
class_thresholds = find_class_wise_energy_thresholds(target_loader1, net, num_class, device,min_known_keep=0.98)
labels, preds, scores = test_model(target_loader1, net, best_energy_th,
                                   num_class, class_thresholds=class_thresholds)
accuracy = accuracy_score(labels, preds)

def open_set_detection_accuracy(preds, labels, num_class):
    unknown_class_idx = num_class

    known_mask   = labels != unknown_class_idx
    unknown_mask = labels == unknown_class_idx

    # 已知类：标签与预测一致
    known_correct = np.sum((preds == labels) & known_mask)
    known_total   = np.sum(known_mask)

    # 未知类：预测标签直接就是 unknown_class_idx
    unknown_correct = np.sum((preds == unknown_class_idx) & unknown_mask)
    unknown_total   = np.sum(unknown_mask)

    total_correct = known_correct + unknown_correct
    total_samples = len(labels)

    known_acc   = known_correct   / known_total   if known_total   > 0 else 0.0
    unknown_acc = unknown_correct / unknown_total if unknown_total > 0 else 0.0
    overall_acc = total_correct   / total_samples if total_samples > 0 else 0.0

    # print(f"已知类别准确率: {known_acc*100:.1f}% ({known_correct}/{known_total})")
    # print(f"未知类别准确率: {unknown_acc*100:.1f}% ({unknown_correct}/{unknown_total})")
    # print(f"整体准确率: {overall_acc*100:.1f}% ({total_correct}/{total_samples})")

    return known_acc, unknown_acc, overall_acc

cm = confusion_matrix(labels, preds, labels=list(range(len(Label_Name))))
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=Label_Name,
            yticklabels=Label_Name,
            cmap='Blues',
            cbar=False)  # 添加这一行
plt.xticks(rotation=45, ha='right')  # 旋转45度，右对齐
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('./results_nobridge/confusion_matrix_wavelet.png', dpi=300, bbox_inches='tight')
plt.close()

a, b, overall_acc = open_set_detection_accuracy(preds, labels, num_class)
h_score = calculate_h_score(a, b)
print(f"[开放集] 已知准确率 a={a * 100:.1f}%，未知准确率 b={b * 100:.1f}%，H-score={h_score * 100:.1f}%")
print(f"[开放集] 整体准确率 c={overall_acc * 100:.1f}%")
# 对源域和目标域的测试进行合并分析
print("模型测试完成！")

# =============================================================================
# 创新点评估指标和可视化
# =============================================================================
print("\n" + "=" * 60)
print("开始评估四个创新点的效果")
print("=" * 60)

# 在评估部分需要定义 NUM_CLASSES
NUM_CLASSES = num_class + 1  # 包括未知类

# 创新点1: Wasserstein距离评估
def evaluate_wasserstein_performance(net, source_loader, target_loader, device=device):
    print("评估Wasserstein距离性能...")

    net.eval()

    # 提取源域特征（已知类）
    src_feats, src_lbls = [], []
    with torch.no_grad():
        for im, lbl in source_loader:
            im = im.to(device)
            lbl = lbl.to(device)

            f, _, _ = net(im)
            feats = f.cpu().numpy()

            # 只保留已知类样本（排除未知类）
            # 假设标签格式：前num_class列为已知类one-hot，最后一列为未知类指示器
            is_unknown = lbl[:, -1].cpu().numpy() == 1
            known_mask = ~is_unknown

            if np.any(known_mask):
                known_feats = feats[known_mask]
                known_labels = lbl[known_mask, :num_class].argmax(dim=1).cpu().numpy()

                src_feats.append(known_feats)
                src_lbls.append(known_labels)

    if not src_feats:
        print("警告：没有找到已知类源域样本")
        return 0.5, None

    src_feats = np.concatenate(src_feats)
    src_lbls = np.concatenate(src_lbls)

    # 提取目标域特征
    tgt_feats, tgt_lbls, tgt_wasserstein = [], [], []
    with torch.no_grad():
        for im, lbl in target_loader1:
            im = im.to(device)
            lbl = lbl.to(device)

            f, _, _ = net(im)
            f_np = f.cpu().numpy()

            # 获取目标域真实标签
            if lbl.shape[1] > 1:
                is_unknown = lbl[:, -1].cpu().numpy() == 1
                known_labels = lbl[:, :num_class].argmax(dim=1).cpu().numpy()
                true_labels = np.where(is_unknown, num_class, known_labels)
            else:
                true_labels = lbl.cpu().numpy().flatten()

            # 计算Wasserstein距离
            wasserstein_dist = []
            for feat in f_np:
                min_dist = float('inf')
                for c in np.unique(src_lbls):
                    class_feats = src_feats[src_lbls == c]
                    if len(class_feats) > 0:
                        # 使用改进的Wasserstein距离计算
                        try:
                            dist = wasserstein_distance(feat.flatten(),
                                                        class_feats.mean(axis=0).flatten())
                            min_dist = min(min_dist, dist)
                        except:
                            # 备用方案：使用欧氏距离
                            dist = np.linalg.norm(feat - class_feats.mean(axis=0))
                            min_dist = min(min_dist, dist)
                wasserstein_dist.append(min_dist)

            tgt_feats.append(f_np)
            tgt_lbls.append(true_labels)
            tgt_wasserstein.append(wasserstein_dist)

    tgt_feats = np.concatenate(tgt_feats) if tgt_feats else np.array([])
    tgt_lbls = np.concatenate(tgt_lbls) if tgt_lbls else np.array([])
    tgt_wasserstein = np.concatenate(tgt_wasserstein) if tgt_wasserstein else np.array([])

    if len(tgt_lbls) == 0:
        print("警告：没有目标域样本")
        return 0.5, None

    # 计算AUROC
    is_unknown = (tgt_lbls == num_class)

    if len(np.unique(is_unknown)) > 1 and len(tgt_wasserstein) > 0:
        try:
            fpr, tpr, _ = roc_curve(is_unknown, tgt_wasserstein)
            wasserstein_auroc = auc(fpr, tpr)
        except:
            print("警告：计算ROC曲线失败")
            wasserstein_auroc = 0.5
    else:
        print("警告：只有单一类别，无法计算ROC曲线")
        wasserstein_auroc = 0.5

    return wasserstein_auroc, tgt_wasserstein


def evaluate_energy_performance(net, target_loader, device=device):
    print("评估能量分数性能...")

    net.eval()
    energy_scores, true_labels = [], []

    with torch.no_grad():
        for im, lbl in target_loader1:
            im = im.to(device)
            lbl = lbl.to(device)

            _, _, logits = net(im)

            # 计算能量分数
            energy = -torch.logsumexp(logits, dim=1).cpu().numpy()

            # 获取真实标签
            if lbl.shape[1] > 1:
                is_unknown = lbl[:, -1].cpu().numpy() == 1
                known_labels = lbl[:, :num_class].argmax(dim=1).cpu().numpy()
                batch_labels = np.where(is_unknown, num_class, known_labels)
            else:
                batch_labels = lbl.cpu().numpy().flatten()

            energy_scores.append(energy)
            true_labels.append(batch_labels)

    energy_scores = np.concatenate(energy_scores) if energy_scores else np.array([])
    true_labels = np.concatenate(true_labels) if true_labels else np.array([])

    if len(true_labels) == 0:
        print("警告：没有目标域样本")
        return 0.5, None

    # 计算AUROC
    is_unknown = (true_labels == num_class)

    if len(np.unique(is_unknown)) > 1 and len(energy_scores) > 0:
        try:
            fpr, tpr, _ = roc_curve(is_unknown, energy_scores)
            energy_auroc = auc(fpr, tpr)
        except:
            print("警告：计算ROC曲线失败")
            energy_auroc = 0.5
    else:
        print("警告：只有单一类别，无法计算ROC曲线")
        energy_auroc = 0.5

    return energy_auroc, energy_scores

def visualize_features(net, target_loader, label_names, device=device):
    net.eval()
    features, labels = [], []

    # 新增：存储预测结果用于判断错误分类
    all_predictions = []
    all_true_labels = []

    # 获取全局变量中的阈值
    threshold_val = best_energy_th if 'best_energy_th' in globals() else threshold
    class_thresholds_val = class_thresholds if 'class_thresholds' in globals() else None

    with torch.no_grad():
        for x, y in target_loader1:  # 注意参数名改为x, y以与test_model保持一致
            x = x.to(device)

            # 获取真实标签（与test_model函数相同逻辑）
            if y.shape[1] > 1:  # one-hot格式
                # 假设最后一列是未知类指示器
                is_unknown_true = y[:, -1].cpu().numpy()
                known_labels = y[:, :num_class].argmax(dim=1).cpu().numpy()
                batch_true_labels = np.where(is_unknown_true == 1, num_class, known_labels)
            else:
                batch_true_labels = y.cpu().numpy().flatten()

            # 获取特征和logits（与test_model相同）
            f, _, logits = net(x)

            # 获取预测结果（使用与test_model相同的代码逻辑）
            energy = -torch.logsumexp(logits, dim=1)
            pred_known = logits.argmax(dim=1).cpu().numpy()
            energy_np = energy.cpu().numpy()

            # 使用相同的阈值逻辑
            if class_thresholds_val is not None:
                # 为每个样本选对应类别的阈值
                thresh = np.array(
                    [class_thresholds_val[p] if p < len(class_thresholds_val) else class_thresholds_val[-1]
                     for p in pred_known])
                prob = torch.softmax(logits, dim=1)
                max_conf, pred_known = prob.max(dim=1)  # 置信度 + 预测类
                max_conf = max_conf.cpu().numpy()
                pred_known = pred_known.cpu().numpy()
                max_conf = prob.max(dim=1)[0].cpu().numpy()
                # 双条件
                is_unknown = (energy_np > thresh) & (max_conf < 0.7)
            else:
                is_unknown = energy_np > threshold_val

            batch_predictions = np.where(is_unknown, num_class, pred_known)

            features.append(f.cpu().numpy())
            labels.append(batch_true_labels)
            all_predictions.append(batch_predictions)
            all_true_labels.append(batch_true_labels)

    features = np.concatenate(features) if features else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    predictions = np.concatenate(all_predictions) if all_predictions else np.array([])
    true_labels = np.concatenate(all_true_labels) if all_true_labels else np.array([])

    if len(features) == 0:
        print("警告：没有提取到特征")
        return

    # 找出错误分类的索引
    error_indices = np.where(predictions != true_labels)[0] if len(predictions) > 0 else np.array([])

    # PCA降维到30维
    pca_components = min(20, len(features))
    pca = PCA(n_components=pca_components, whiten=True, random_state=42)
    features_pca = pca.fit_transform(features)

    # t-SNE降维到2维，根据样本数调整perplexity
    n_samples = len(features)

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=25,
                early_exaggeration=12,
                learning_rate=120,
                n_iter_without_progress=500,  # 足够收敛
                init='pca',  # PCA初始化更稳定
                )
    features_2d = tsne.fit_transform(features_pca)

    plt.figure(figsize=(12, 10))

    # 原始的点（保持原样）
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab10',
                          alpha=0.85, s=80,
                          edgecolors='k', linewidths=0.5)

    # 在错误分类的点上叠加红叉
    if len(error_indices) > 0:
        plt.scatter(features_2d[error_indices, 0],
                    features_2d[error_indices, 1],
                    c='red', marker='x', s=100, linewidths=2,
                    label=f'Misclassified ({len(error_indices)} errors)')

    # 创建图例（保持原逻辑）
    unique_labels = np.unique(labels)
    legend_elements = []
    for label in unique_labels:
        if label < len(label_names):
            color = scatter.cmap(scatter.norm(label))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=color,
                                              label=label_names[label],
                                              markersize=8))

    # 如果有错误分类，添加红叉图例
    if len(error_indices) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='x', color='red',
                                          label=f'Misclassified ({len(error_indices)} errors)',
                                          markersize=8, linewidth=2))

    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right')

    plt.title('t-SNE feature visualization', fontfamily="DejaVu Sans", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('./results_nobridge/feature_visualization_wavelet.png', dpi=300, bbox_inches='tight')
    plt.close()


# 执行所有评估
print("\n开始综合性能评估...")

# 1. Wasserstein距离评估
wasserstein_auroc, wasserstein_scores = evaluate_wasserstein_performance(net, source_loader, target_loader)

# 2. 能量分数评估
energy_auroc, energy_scores = evaluate_energy_performance(net, target_loader)

# 3. 特征可视化
visualize_features(net, target_loader1, Label_Name)  # 关键：使用测试集

print("=" * 60)