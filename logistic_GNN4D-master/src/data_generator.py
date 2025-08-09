import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch import optim
import torch.nn.functional as F
from load import get_P, get_Pd, get_W_lg
import random
from controlsnr import find_a_given_snr
from scipy.sparse import csr_matrix, save_npz

def simple_collate_fn(batch):
    """
    拼接 batch，适用于所有图大小一致的情况。
    返回：
      - adj: [B, N, N]
      - labels: [B, N]
    """
    adjs = [torch.tensor(sample['adj'].toarray(), dtype=torch.float32) for sample in batch]
    labels = [torch.tensor(sample['labels'], dtype=torch.long) for sample in batch]

    adj_batch = torch.stack(adjs)       # [B, N, N]
    label_batch = torch.stack(labels)   # [B, N]

    return {
        'adj': adj_batch,
        'labels': label_batch
    }

class Generator(object):
    def __init__(self, N_train=50, N_test=100, N_val = 50,generative_model='SBM_multiclass', p_SBM=0.8, q_SBM=0.2, n_classes=2, path_dataset='dataset',
                 num_examples_train=100, num_examples_test=10, num_examples_val=10):
        self.N_train = N_train
        self.N_test = N_test
        self.N_val = N_val

        self.generative_model = generative_model
        self.p_SBM = p_SBM
        self.q_SBM = q_SBM
        self.n_classes = n_classes
        self.path_dataset = path_dataset

        self.data_train = None
        self.data_test = None
        self.data_val = None

        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test
        self.num_examples_val = num_examples_val

        self.fixed_class_sizes = [
            (500, 500),
            (400, 600),
            (300, 700),
            (200, 800),
            (100, 900),
            (50, 950)
        ]

    def SBM(self, p, q, N):
        W = np.zeros((N, N))

        p_prime = p
        q_prime = q

        n = N // 2

        W[:n, :n] = np.random.binomial(1, p, (n, n))
        W[n:, n:] = np.random.binomial(1, p, (N-n, N-n))
        W[:n, n:] = np.random.binomial(1, q, (n, N-n))
        W[n:, :n] = np.random.binomial(1, q, (N-n, n))
        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        blockA = perm < n
        labels = blockA * 2 - 1

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels


    def SBM_multiclass(self, p, q, N, n_classes):

        p_prime = p
        q_prime = q

        prob_mat = np.ones((N, N)) * q_prime

        n = N // n_classes  # 基础类别大小
        remainder = N % n_classes  # 不能整除的剩余部分
        n_last = n + remainder  # 最后一类的大小

        # 先对整除部分进行块状分配
        for i in range(n_classes - 1):  # 处理前 n_classes-1 类
            prob_mat[i * n: (i + 1) * n, i * n: (i + 1) * n] = p_prime

        # 处理最后一类
        start_idx = (n_classes - 1) * n  # 最后一类的起始索引
        prob_mat[start_idx: start_idx + n_last, start_idx: start_idx + n_last] = p_prime

        # 生成邻接矩阵
        W = np.random.rand(N, N) < prob_mat
        W = W.astype(int)

        W = W * (np.ones(N) - np.eye(N))  # 移除自环
        W = np.maximum(W, W.transpose())  # 确保无向图

        # 随机打乱节点顺序
        perm = torch.randperm(N).numpy()

        # 生成类别标签
        labels =np.minimum((perm // n) , n_classes - 1)

        W_permed = W[perm]
        W_permed = W_permed[:, perm]

        #计算P矩阵的特征向量
        prob_mat_permed = prob_mat[perm][:, perm]
        # np.fill_diagonal(prob_mat_permed, 0)  # 去除自环

        eigvals, eigvecs = np.linalg.eigh(prob_mat_permed)
        idx = np.argsort(eigvals)[::-1]
        eigvecs_top = eigvecs[:, idx[:n_classes]]

        return W_permed, labels, eigvecs_top  # 返回前n_classes特征向量


    def imbalanced_SBM_multiclass(self, p, q, N, n_classes, class_sizes):

        p_prime = p
        q_prime = q

        prob_mat = np.ones((N, N)) * q_prime

        # 计算类别区间的索引边界
        boundaries = np.cumsum([0] + class_sizes)

        for i in range(n_classes):
            start = boundaries[i]
            end = boundaries[i + 1]
            prob_mat[start:end, start:end] = p_prime

        W = np.random.rand(N, N) < prob_mat
        W = W.astype(int)

        W = W * (np.ones((N, N)) - np.eye(N))
        W = np.maximum(W, W.T)

        # 打乱节点顺序
        perm = torch.randperm(N).numpy()

        # 根据 perm，重新分配 labels
        labels = np.zeros(N, dtype=int)
        for i in range(n_classes):
            start = boundaries[i]
            end = boundaries[i + 1]
            labels[start:end] = i

        labels = labels[perm]

        W_permed = W[perm, :]
        W_permed = W_permed[:, perm]

        #计算P矩阵的特征向量
        prob_mat_permed = prob_mat[perm][:, perm]
        # np.fill_diagonal(prob_mat_permed, 0)  # 去除自环

        eigvals, eigvecs = np.linalg.eigh(prob_mat_permed)
        idx = np.argsort(eigvals)[::-1]
        eigvecs_top = eigvecs[:, idx[:n_classes]]

        return W_permed, labels, eigvecs_top  # 返回前n_classes特征向量

    def create_dataset_random_otf(self, directory, mode='train', C=10, min_size=50):
        """
        生成随机 SBM_multiclass 图（不使用固定 p/q），并逐图保存为稀疏格式 .npz 文件。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if mode == 'train':
            graph_size = self.N_train
            num_graphs = self.num_examples_train
        elif mode == 'test':
            graph_size = self.N_test
            num_graphs = self.num_examples_test
        elif mode == 'val':
            graph_size = self.N_val
            num_graphs = self.num_examples_val
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        for i in range(num_graphs):
            # Step 1: SNR 控制边密度
            a_low, b_low = find_a_given_snr(0.1, self.n_classes, C)
            a_high, b_high = find_a_given_snr(1, self.n_classes, C)
            lower_bound = a_low / b_low
            upper_bound = a_high / b_high

            # Step 2: 生成 SBM 参数和图
            p, q, class_sizes, snr = self.random_imbalanced_SBM_generator_balanced_sampling(
                N=graph_size,
                n_classes=self.n_classes,
                C=C,
                alpha_range=(lower_bound, upper_bound),
                min_size=min_size
            )

            W_dense, labels, eigvecs_top = self.SBM_multiclass(
                p, q, graph_size, self.n_classes
            )

            # Step 3: 稀疏化邻接矩阵
            W_sparse = csr_matrix(W_dense)

            # Step 4: 保存为稀疏 .npz 格式
            graph_path = os.path.join(directory, f"graph_{i:04d}.npz")
            np.savez_compressed(
                graph_path,
                adj_data=W_sparse.data,
                adj_indices=W_sparse.indices,
                adj_indptr=W_sparse.indptr,
                adj_shape=W_sparse.shape,
                labels=labels,
                p=p,
                q=q,
                snr=snr,
                class_sizes=np.array(class_sizes)
            )

        print(f" {mode} 数据集已保存到目录: {directory}")

        # 不再加载所有图进内存
        if mode == 'train':
            self.data_train = directory
        elif mode == 'test':
            self.data_test = directory
        elif mode == 'val':
            self.data_val = directory

    def prepare_data(self):
        def get_npz_dataset(path, mode):
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"[创建数据集] {mode} 数据目录不存在，已新建：{path}")

            npz_files = sorted([f for f in os.listdir(path) if f.endswith(".npz")])
            if not npz_files:
                print(f"[创建数据集] {mode} 数据未找到，开始生成...")
                self.create_dataset_random_otf(path, mode=mode)
                npz_files = sorted([f for f in os.listdir(path) if f.endswith(".npz")])
            else:
                print(f"[读取数据] {mode} 集已存在，共 {len(npz_files)} 张图：{path}")

            # 返回路径列表
            return [os.path.join(path, f) for f in npz_files]

        train_dir = f"{self.generative_model}_nc{self.n_classes}_rand_gstr{self.N_train}_numtr{self.num_examples_train}"
        test_dir = f"{self.generative_model}_nc{self.n_classes}_rand_gste{self.N_test}_numte{self.num_examples_test}"
        val_dir = f"{self.generative_model}_nc{self.n_classes}_rand_val{self.N_val}_numval{self.num_examples_val}"

        train_path = os.path.join(self.path_dataset, train_dir)
        test_path = os.path.join(self.path_dataset, test_dir)
        val_path = os.path.join(self.path_dataset, val_dir)

        self.data_train = get_npz_dataset(train_path, 'train')
        self.data_test = get_npz_dataset(test_path, 'test')
        self.data_val = get_npz_dataset(val_path, 'val')


    def sample_single(self, i, is_training=True):
        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        example = dataset[i]
        if (self.generative_model == 'SBM_multiclass'):
            W_np = example['W']
            labels = np.expand_dims(example['labels'], 0)
            labels_var = torch.from_numpy(labels)
            if is_training:
                labels_var.requires_grad = True
            return W_np, labels_var


    def sample_otf_single(self, is_training=True, cuda=True):
        if is_training:
            N = self.N_train
        else:
            N = self.N_test
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, N)
        elif self.generative_model == 'SBM_multiclass':
            W, labels,eigvecs_top = self.SBM_multiclass(self.p_SBM, self.q_SBM, N, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))

        labels = np.expand_dims(labels, 0)
        labels = torch.from_numpy(labels)
        W = np.expand_dims(W, 0)
        # W = torch.tensor(W, dtype=torch.float32)  # 不加 requires_grad

        return W, labels, eigvecs_top

    def imbalanced_sample_otf_single(self, class_sizes , is_training=True, cuda=True):
        if is_training:
            N = self.N_train
        else:
            N = self.N_test
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, N)
        elif self.generative_model == 'SBM_multiclass':
            W, labels,eigvecs_top = self.imbalanced_SBM_multiclass(self.p_SBM, self.q_SBM, N, self.n_classes, class_sizes)
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))

        labels = np.expand_dims(labels, 0)
        labels = torch.from_numpy(labels)
        W = np.expand_dims(W, 0)
        # W = torch.tensor(W, dtype=torch.float32)  # 不加 requires_grad

        return W, labels, eigvecs_top


    def random_sample_otf_single(self, C = 10 ,is_training=True, cuda=True):
        if is_training:
            N = self.N_train
        else:
            N = self.N_test
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, N)

        elif self.generative_model == 'SBM_multiclass':
            a_low, b_low = find_a_given_snr(0.1, self.n_classes, C)
            a_high, b_high = find_a_given_snr(1, self.n_classes, C)

            lower_bound = a_low / b_low
            upper_bound = a_high / b_high

            if lower_bound > upper_bound:
                lower_bound, upper_bound = upper_bound, lower_bound

            p, q, class_sizes, snr = self.random_imbalanced_SBM_generator_balanced_sampling(
                N=N,
                n_classes=self.n_classes,
                C=C,
                alpha_range=(lower_bound, upper_bound),
                min_size= 20
            )
            W, labels,eigvecs_top = self.imbalanced_SBM_multiclass(p, q, N, self.n_classes, class_sizes)

        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))

        labels = np.expand_dims(labels, 0)
        labels = torch.from_numpy(labels)
        W = np.expand_dims(W, 0)
        # W = torch.tensor(W, dtype=torch.float32)  # 不加 requires_grad

        return W, labels, eigvecs_top, snr, class_sizes


    def random_imbalanced_SBM_generator_balanced_sampling(self, N, n_classes, C, *,
                                        alpha_range=(1.3, 2.8),
                                        min_size=5):
        """
        随机生成 SBM 模型的参数，社区大小为随机比例但总和为 N。
        返回 p, q, class_sizes, a, b, snr。
        """
        assert N >= min_size * n_classes

        # Step 1: 随机生成 a > b，使得 a + (k - 1) * b = C
        alpha = np.random.uniform(*alpha_range)
        b = C / (alpha + (n_classes - 1))
        a = alpha * b

        # Step 2: 计算边连接概率
        logn = np.log(N)
        p = a * logn / N
        q = b * logn / N

        # ✅ Step 3: 使用 Dirichlet 生成 class_sizes
        remaining = N - min_size * n_classes
        probs = np.random.dirichlet(np.ones(n_classes))  # 总和为1的概率向量
        extras = np.random.multinomial(remaining, probs)
        class_sizes = [min_size + e for e in extras]

        # Step 4: 计算 SNR
        snr = (a - b) ** 2 / (n_classes * (a + (n_classes - 1) * b))

        return p, q, class_sizes, snr

    def copy(self):
        return copy.deepcopy(self)