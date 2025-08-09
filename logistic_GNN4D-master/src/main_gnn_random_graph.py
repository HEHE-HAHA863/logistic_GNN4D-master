import os
from multiprocessing import Pool

import multiprocessing
from torch.utils.data import Dataset
import openpyxl
from data_generator import Generator, simple_collate_fn
from load import get_gnn_inputs
from models import GNN_multiclass, GNN_multiclass_second_period
import time
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch import Tensor
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from get_infor_from_first_step import get_second_period_labels_single, test_first_get_second_period_labels_single, in_sample_test_first_get_second_period_labels_single, imbalanced_test_first_get_second_period_labels_single
from losses import compute_loss_multiclass, compute_accuracy_multiclass, compute_accuracy_spectral
from train_first_period import  train_first_period_with_early_stopping
from load_local_refinement import get_gnn_inputs_local_refinement
from controlsnr import  find_a_given_snr
import sys
import os
import time
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz

def setup_logger(prefix="main_gnn"):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    log_filename = f"{prefix}_{timestamp}_{pid}.log"
    logfile = open(log_filename, "w", buffering=1)  # 行缓冲
    sys.stdout = logfile
    sys.stderr = logfile
    print(f"[Logger initialized] Logging to: {log_filename}")


parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
#                          提前配置参数，方便后面使用                              #
###############################################################################

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(6000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(100))
parser.add_argument('--num_examples_val', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--p_SBM', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.1)
parser.add_argument('--class_sizes', type=int, nargs='+', default=[100, 1000],
                    help='List of class sizes for imbalanced SBM')
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=2)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
#########################
#parser.add_argument('--generative_model', nargs='?', const=1, type=str,
#                    default='ErdosRenyi')
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='SBM_multiclass')
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default= 16)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='test')
default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--mode_isbalanced', nargs='?', const=1, type=str, default='imbalanced')
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default=default_path)
parser.add_argument('--path_local_refinement', nargs='?', const=1, type=str, default='')

parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--filename_existing_gnn_local_refinement', nargs='?', const=1, type=str, default='')

parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=1)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)
parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
parser.set_defaults(eval_vs_train=True)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=8)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=50)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default=2)
parser.add_argument('--J', nargs='?', const=1, type=int, default= 2)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N_val', nargs='?', const=1, type=int, default=1000)

parser.add_argument('--lr', nargs='?', const=1, type=float, default=4e-3)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10}'
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} '
template4 = '{:<10} {:<10.5f} {:<10.5f} \n'

template_header = '{:<6} {:<10} {:<10} {:<13} {:<10} {:<8} {:<10} {:<10} {:<20}'
template_row    = '{:<6} {:<10.5f} {:<10.5f} {:<13} {:<10} {:<8} {:<10.3f} {:<10.4f} {:<20}'



class SBMDataset(torch.utils.data.Dataset):
    def __init__(self, npz_file_list):
        self.files = npz_file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        adj = csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), shape=tuple(data['adj_shape']))
        labels = data['labels']
        return {'adj': adj, 'labels': labels, 'num_nodes': adj.shape[0]}

import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian

def to_one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将整数标签转为独热编码，返回 (N, k) 的矩阵
    """
    N = labels.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), labels] = 1
    return one_hot

def local_refinement_by_neighbors(A: np.ndarray, pred_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    基于邻接矩阵 A 的 local refinement：将每个点重新归属到与其连接边最多的社区。

    A: (N, N) 邻接矩阵
    pred_labels: (N,) 初始预测标签
    num_classes: 社区数

    return: (N,) 精炼后的预测标签
    """
    one_hot = to_one_hot(pred_labels, num_classes)  # (N, k)
    community_scores = A @ one_hot                  # (N, k)
    refined_labels = np.argmax(community_scores, axis=1)  # (N,)
    return refined_labels

def spectral_clustering_adj(A: np.ndarray, k: int, true_labels, normalized: bool = True):
    """
    对邻接矩阵 A 做谱聚类 + local refinement。

    返回：
    - acc_spectral_clustering: 初始谱聚类准确率
    - acc_refined: refinement 后准确率
    - refined_pred: refinement 后标签
    """
    A = A.squeeze(0)  # (N, N)

    # 拉普拉斯矩阵
    L = laplacian(A, normed=normalized)

    # 特征分解
    eigvals, eigvecs = eigh(L, subset_by_index=(0, k - 1))
    U = eigvecs
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)  # 行归一化

    # KMeans 聚类
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(U_norm)

    # 匹配标签并计算准确率
    acc_spectral_clustering, best_matched_pred = compute_accuracy_spectral(labels, true_labels, k)

    # Local refinement
    refined_pred = local_refinement_by_neighbors(A, best_matched_pred, k)
    acc_refined, _ = compute_accuracy_spectral(refined_pred, true_labels, k)

    return acc_spectral_clustering, acc_refined

def get_available_device():
    for i in range(torch.cuda.device_count()):
        try:
            # 尝试分配临时张量测试显存
            torch.cuda.set_device(i)
            torch.zeros(1).cuda()
            return torch.device(f"cuda:{i}")
        except RuntimeError:
            continue
    return torch.device("cpu")


device = get_available_device()


# ##############Get the labels from the first period and use this labels to train the local refinement###################
# def train_single_local_refinement(gnn_first_period, gnn_local_refine, n_classes, W, true_labels ,optimizer, iter):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     sam_com_matrix, pred_labels = get_second_period_labels_single(gnn_first_period, W, true_labels,n_classes, args)
#
#     true_labels = true_labels.type(dtype_l)
#
#     start = time.time()
#     WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_labels, n_classes)
#
#     # **移动数据到计算设备 (GPU/CPU)**
#     WW = WW.to(device)
#     x = x.to(device)
#
#     optimizer.zero_grad(set_to_none=True)
#     pred = gnn_local_refine(WW.type(dtype), x.type(dtype))
#
#     loss = compute_loss_multiclass(pred, true_labels, n_classes)  # 计算损失
#     loss.backward()
#     nn.utils.clip_grad_norm_(gnn_local_refine.parameters(), args.clip_grad_norm)
#     optimizer.step()
#
#     acc, best_matched_pred = compute_accuracy_multiclass(pred, true_labels, n_classes)
#
#     elapsed_time = time.time() - start
#
#     if torch.cuda.is_available():
#         loss_value = float(loss.data.cpu().numpy())
#     else:
#         loss_value = float(loss.data.numpy())
#
#     info = ['iter', 'avg loss', 'avg acc', 'edge_density',
#             'noise', 'model', 'elapsed']
#     out = [iter, loss_value, acc, args.edge_density,
#            args.noise, 'GNN', elapsed_time]
#
#     print(template1.format(*info))
#     print(template2.format(*out))
#
#     # **释放 GPU 显存**
#     WW = None
#     x = None
#
#     return loss_value, acc
#
#
# def train_local_refinement(gnn_first_period, gnn_local_refine, n_classes=args.n_classes, iters=args.num_examples_train):
#     gnn_local_refine.train()
#     optimizer = torch.optim.Adamax(gnn_local_refine.parameters(), lr=args.lr)
#
#     loss_lst = np.zeros([iters])
#     acc_lst = np.zeros([iters])
#
#     for it in range(iters):
#         W_i = cached_graphs[it]
#         true_labels_i = cached_labels[it]
#
#         loss_single, acc_single = train_single_local_refinement(gnn_first_period, gnn_local_refine, n_classes, W_i, true_labels_i,
#                                                                 optimizer, it)
#
#         loss_lst[it] = loss_single
#         acc_lst[it] = acc_single
#
#         torch.cuda.empty_cache()

# def get_insample_acc_lost_single(gnn_first_period, gnn_local_refine, W_i, labels_i, n_classes, args):
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     W, true_labels = W_i, labels_i
#
#     sam_com_matrix, pred_label_first ,first_in_sample_loss, first_in_sample_acc \
#      = in_sample_test_first_get_second_period_labels_single(gnn_first_period, W, true_labels,n_classes, args)
#
#     WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_label_first, n_classes)
#
#     if torch.cuda.is_available():
#         WW = WW.to(device)
#         x = x.to(device)
#
#     pred_single_second = gnn_local_refine(WW.type(dtype), x.type(dtype))
#
#     # 计算第二阶段损失和准确率
#     second_in_sample_loss = compute_loss_multiclass(pred_single_second, true_labels, n_classes)
#     second_in_sample_acc, best_matched_pred = compute_accuracy_multiclass(pred_single_second, true_labels, n_classes)
#
#     WW = None
#     x = None
#
#     return first_in_sample_loss, first_in_sample_acc, float(second_in_sample_loss.data.cpu().numpy()), second_in_sample_acc
#
# def get_insample_acc_lost(gnn_first_period, gnn_local_refinement, n_classes, iters=args.num_examples_test,
#                           filename="in_sample_test_results_sparsity.csv"):
#
#     gnn_first_period.train()
#     gnn_local_refinement.train()
#
#     # in_sample_loss_lst_first = np.zeros([iters])
#     in_sample_acc_lst_first = np.zeros([iters])
#
#     # in_sample_loss_lst_second = np.zeros([iters])
#     in_sample_acc_lst_second = np.zeros([iters])
#
#     for it in range(iters):
#         W_i = cached_graphs[it]
#         labels_i = cached_labels[it]
#
#         first_in_sample_loss, first_in_sample_acc, second_in_sample_loss, second_in_sample_acc = get_insample_acc_lost_single(
#             gnn_first_period,
#             gnn_local_refinement, W_i, labels_i, n_classes, args)
#
#         in_sample_acc_lst_first[it] = first_in_sample_acc
#
#         in_sample_acc_lst_second[it] = second_in_sample_acc
#
#         torch.cuda.empty_cache()
#     # 计算均值和标准差
#     first_avg_test_acc = np.mean(in_sample_acc_lst_first)
#     first_std_test_acc = np.std(in_sample_acc_lst_first)
#
#     second_avg_test_acc = np.mean(in_sample_acc_lst_second)
#     second_std_test_acc = np.std(in_sample_acc_lst_second)
#
#     n = args.N_train  # 或者 N_test，也可以统一都用 N
#     logn_div_n = np.log(n) / n
#
#     a = args.p_SBM / logn_div_n
#     b = args.q_SBM / logn_div_n
#     k = args.n_classes
#
#     snr = (a - b) ** 2 / (k * (a + (k - 1) * b))
#
#     df = pd.DataFrame([{
#         "n_classes": args.n_classes,
#         "p_SBM": args.p_SBM,
#         "q_SBM": args.q_SBM,
#         "J": args.J,
#         "N_train": args.N_train,
#         "N_test": args.N_test,
#         "first_avg_test_acc": first_avg_test_acc,
#         "first_std_test_acc": first_std_test_acc,
#         "second_avg_test_acc": second_avg_test_acc,
#         "second_std_test_acc": second_std_test_acc,
#         "SNR": snr
#     }])
#
#     # 追加模式写入文件，防止覆盖
#     df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))

def test_single_first_period(gnn_first_period, gen, n_classes, args, iter, mode='balanced', class_sizes=None):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_first_period.train()

    # 选择模式
    if mode == 'imbalanced':
        W, true_labels, eigvecs_top = gen.imbalanced_sample_otf_single(class_sizes, is_training=True, cuda=True)
        true_labels = true_labels.type(dtype_l)

    acc_spectral_clustering, acc_refined = spectral_clustering_adj(W, n_classes, true_labels)

    # GNN 输入
    WW, x = get_gnn_inputs(W, args.J)
    WW, x = WW.to(device), x.to(device)

    # 禁用梯度计算
    with torch.no_grad():
        pred_single_first = gnn_first_period(WW.type(dtype), x.type(dtype))

    # --- 3. 计算邻接矩阵特征向量 ---
    W_np = W.squeeze(0).cpu().numpy() if isinstance(W, Tensor) else W.squeeze(0)

    W_for_eig = (W_np + W_np.T) / 2  # 确保对称
    eigvals_W, eigvecs_W = np.linalg.eigh(W_for_eig)
    eigvals_W, eigvecs_W = np.real(eigvals_W), np.real(eigvecs_W)
    adjacency_eigvecs = eigvecs_W[:, np.argsort(eigvals_W)[-n_classes:][::-1]]
    adjacency_eigvecs /= np.linalg.norm(adjacency_eigvecs, axis=0, keepdims=True)

    # 中间层特征
    penultimate_features = gnn_first_period.get_penultimate_output().detach().cpu().numpy().squeeze(0)
    penultimate_features /= np.linalg.norm(penultimate_features, axis=0, keepdims=True)

    # 第二阶段 loss 和 acc
    loss_test_first = compute_loss_multiclass(pred_single_first, true_labels, n_classes)
    acc_test_first, best_matched_pred = compute_accuracy_multiclass(pred_single_first, true_labels, n_classes)
    pred_label_first = best_matched_pred

    N = true_labels.shape[1]
    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss_test_first.data.cpu().numpy())
    else:
        loss_value = float(loss_test_first.data.numpy())

    info = ['iter', 'avg loss', 'avg acc', 'edge_density',
            'noise', 'model', 'elapsed']
    out = [iter, loss_value, acc_test_first, args.edge_density,
           args.noise, 'GNN', elapsed]
    print(template1.format(*info))
    print(template2.format(*out))

    del WW
    del x

    # 构造 Excel 表
    data = {
        'True_Label': true_labels.squeeze(0).cpu().numpy(),
        'Pred_Label_First': pred_label_first.reshape(-1),
        'Loss_First': [float(loss_test_first)] * N,
        'Acc_First': [float(acc_test_first)] * N,
    }

    for i in range(2 * n_classes):
        data[f'penultimate_GNN_Feature{i + 1}'] = penultimate_features[:, i]
    for i in range(n_classes):
        data[f'eigvecs_top{i + 1}'] = eigvecs_top[:, i]
    for i in range(n_classes):
        data['Adj_EigVecs_Top' + str(i + 1) + ''] = adjacency_eigvecs[:, i]

    df = pd.DataFrame(data)

    # 写入 Excel
    root_folder = "penultimate_GNN_Feature"
    subfolder_name = f"penultimate_GNN_Feature_nclasses_{n_classes}"
    output_filename = (
        f"first_gnn_classesizes={class_sizes}_p={gen.p_SBM}_q={gen.q_SBM}_"
        f"j={args.J}_nlyr={args.num_layers}.xlsx"
    )
    output_path = os.path.join(root_folder, subfolder_name, output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if iter < 10:
        if iter == 0:
            df.to_excel(output_path, sheet_name=f'Iteration_{iter}', index=False)
        else:
            with pd.ExcelWriter(output_path, mode='a', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=f'Iteration_{iter}', index=False)

    return loss_test_first, acc_test_first ,acc_spectral_clustering, acc_refined


def test_first_period(gnn_first_period, n_classes, gen, iters=args.num_examples_test,
                          mode='balanced', class_sizes=None,
                          filename="random_test_results_sparsity.csv"):

    gnn_first_period.train()

    loss_lst_first = np.zeros([iters])
    acc_lst_first = np.zeros([iters])

    acc_spectral_clustering_list = np.zeros([iters])

    acc_refined_list = np.zeros([iters])

    for it in range(iters):
        loss_test_first, acc_test_first, acc_spectral_clustering, acc_refined = test_single_first_period(
            gnn_first_period, gen, n_classes, args, it, mode, class_sizes)

        loss_lst_first[it] = loss_test_first
        acc_lst_first[it] = acc_test_first

        acc_spectral_clustering_list[it] = acc_spectral_clustering
        acc_refined_list[it] = acc_refined

        torch.cuda.empty_cache()
    # 计算均值和标准差
    first_avg_test_acc = np.mean(acc_lst_first)

    spectral_clustering_avg_test_acc = np.mean(acc_spectral_clustering_list)

    acc_refined_avg_test_acc = np.mean(acc_refined_list)

    n = args.N_train  # 或者 N_test，也可以统一都用 N
    logn_div_n = np.log(n) / n

    a = gen.p_SBM / logn_div_n
    b = gen.q_SBM / logn_div_n
    k = args.n_classes

    snr = (a - b) ** 2 / (k * (a + (k - 1) * b))

    df = pd.DataFrame([{
        "n_classes": args.n_classes,
        "class_sizes": class_sizes,
        "p_SBM": gen.p_SBM,
        "q_SBM": gen.q_SBM,
        "J": args.J,
        "N_train": args.N_train,
        "N_test": args.N_test,
        "first_avg_test_acc": first_avg_test_acc,
        "spectral_clustering_avg_test_acc": spectral_clustering_avg_test_acc,
        "acc_refined_avg_test_acc" : acc_refined_avg_test_acc,
        "SNR": snr,
    }])

    # 追加模式写入文件，防止覆盖
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))


def test_first_period_wrapper(args_tuple):
    gnn_first_period, class_sizes, snr, gen, logN_div_N, total_ab= args_tuple

    a, b = find_a_given_snr(snr, args.n_classes, total_ab)
    p_SBM = round(a * logN_div_N, 4)
    q_SBM = round(b * logN_div_N, 4)

    # 每个进程用自己的 gen 副本（你可以用 copy 或重新初始化）
    gen_local = gen.copy()  # 确保你实现了 Generator.copy() 方法
    gen_local.p_SBM = p_SBM
    gen_local.q_SBM = q_SBM

    print(f"\n[测试阶段] class_sizes: {class_sizes}, SNR: {snr:.2f}")
    print(f"使用的 SBM 参数: p={p_SBM}, q={q_SBM}")

    return test_first_period(
        gnn_first_period=gnn_first_period,  # 这一步如果模型只读是可以的
        n_classes=args.n_classes,
        gen=gen_local,
        iters=args.num_examples_test,
        mode=args.mode_isbalanced,
        class_sizes=class_sizes,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    try:
        setup_logger("run_gnn")
        gen = Generator()
        gen.N_train = args.N_train
        gen.N_test = args.N_test
        gen.N_val = args.N_val

        gen.edge_density = args.edge_density
        gen.p_SBM = args.p_SBM
        gen.q_SBM = args.q_SBM

        gen.random_noise = args.random_noise
        gen.noise = args.noise
        gen.noise_model = args.noise_model
        gen.generative_model = args.generative_model
        gen.n_classes = args.n_classes

        gen.num_examples_train = args.num_examples_train
        gen.num_examples_test = args.num_examples_test
        gen.num_examples_val = args.num_examples_val

        # 1. 创建总模型文件夹
        root_model_dir = "model_GNN"
        os.makedirs(root_model_dir, exist_ok=True)

        # 2. 创建子目录（按 n_classes 分类）
        folder_name = f"GNN_model_first_classes{args.n_classes}"
        full_save_dir = os.path.join(root_model_dir, folder_name)
        os.makedirs(full_save_dir, exist_ok=True)

        # 3. 构造保存路径
        filename_first = (
            f'gnn_J{args.J}_lyr{args.num_layers}_classes{args.n_classes}'
        )
        path_first = os.path.join(full_save_dir, filename_first)

        ################################################################################################################
        # Here is the train period, prepare the dataloader we need to train
        gen.prepare_data()

        # 1. 准备并行线程数
        num_workers = min(4, multiprocessing.cpu_count() - 1)
        print("num_workers", num_workers)

        # 2. 创建 Dataset 实例
        train_dataset = SBMDataset(gen.data_train)
        val_dataset = SBMDataset(gen.data_val)
        test_dataset = SBMDataset(gen.data_test)

        # 3. 创建对应的 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=simple_collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # ❗验证集不需要打乱顺序
            num_workers=num_workers,
            collate_fn=simple_collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=simple_collate_fn

        )

        if args.mode == "train":

            print(
                f"[阶段一] 训练 GNN：n_classes={args.n_classes}, layers={args.num_layers}, J = {args.J}")
            # 初始化并训练第一阶段
            if args.generative_model == 'SBM_multiclass':
                gnn_first_period = GNN_multiclass(args.num_features, args.num_layers, args.J + 3,
                                                  n_classes=args.n_classes)
            if torch.cuda.is_available():
                gnn_first_period = gnn_first_period.to(device)

            loss_list, acc_list = train_first_period_with_early_stopping(
                gnn_first_period, train_loader, val_loader, args.n_classes, args,
                epochs=100, patience=6, save_path = filename_first)


            print(f'Saving first-period GNN to {path_first}')
            torch.save(gnn_first_period.cpu(), path_first)

            if torch.cuda.is_available():
                gnn_first_period = gnn_first_period.to('cuda')

        print("[测试阶段] 开始...")

        gnn_first_period = torch.load(path_first, map_location=torch.device('cpu'))

        print("[测试阶段] 开始...")

        class_sizes_list = [
            [50, 950],

            [100, 900],

            [200, 800],

            [300, 700],

            [400, 600],

            [500, 500]
        ]

        snr_list = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]

        total_ab = 10

        N = 1000

        logN_div_N = np.log(N) / N  # ≈ 0.0069

        # 构造所有任务

        task_args = []

        for class_sizes in class_sizes_list:

            for snr in snr_list:
                task_args.append((gnn_first_period, class_sizes, snr, gen, logN_div_N, total_ab))

        # 并行执行

        print("[测试阶段] 并行执行中...")

        with Pool(nodes=4) as pool:  # nodes 可调节，比如 4~8

            results = pool.map(test_first_period_wrapper, task_args)


    except Exception as e:

        import traceback

        traceback.print_exc()