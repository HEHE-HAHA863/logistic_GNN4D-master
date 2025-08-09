import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import itertools



criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor


def compute_accuracy_spectral(pred, true, n_classes=None):
    """
    计算在最佳标签排列下的准确率和重标记后的预测结果。

    参数:
        pred (np.ndarray): 预测标签，shape = (1, N) 或 (N,)
        true (np.ndarray): 真实标签，shape = (1, N) 或 (N,)
        n_classes (int): 类别数（可自动推断）

    返回:
        best_acc: 最佳排列下的准确率
        best_pred: 对应排列下的预测标签
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()

    pred = pred.flatten()
    true = true.flatten()

    if n_classes is None:
        n_classes = max(np.max(pred), np.max(true)) + 1

    best_acc = -1
    best_pred = pred.copy()

    for perm in itertools.permutations(range(n_classes)):
        mapping = {i: perm[i] for i in range(n_classes)}
        remapped_pred = np.array([mapping[p] for p in pred])
        acc = np.mean(remapped_pred == true)
        if acc > best_acc:
            best_acc = acc
            best_pred = remapped_pred.copy()

    return best_acc, best_pred

def from_scores_to_labels_multiclass_batch(pred):
    labels_pred = np.argmax(pred, axis = 2).astype(int)
    return labels_pred
    # return (1,1000)

def compute_accuracy_multiclass_batch(labels_pred, labels):
    overlap = (labels_pred == labels).astype(int)
    acc = np.mean(labels_pred == labels)
    return acc

def compute_loss_multiclass(pred_llh, labels, n_classes):
    loss = 0
    permutations = permuteposs(n_classes)
    batch_size = pred_llh.data.cpu().shape[0]
    for i in range(batch_size):
        pred_llh_single = pred_llh[i, :, :]
        labels_single = labels[i, :]
        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            labels_under_perm = torch.from_numpy(permutations[j, labels_single.data.cpu().numpy().astype(int)])
            loss_under_perm = criterion(pred_llh_single, labels_under_perm.type(dtype_l))

            if (j == 0):
                loss_single = loss_under_perm
            else:
                loss_single = torch.min(loss_single, loss_under_perm)

        loss += loss_single

    return loss / batch_size


def compute_accuracy_multiclass(pred_llh, labels, n_classes):
    pred_llh = pred_llh.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    batch_size = pred_llh.shape[0]
    pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)
    permutations = permuteposs(n_classes)

    best_matched_preds = np.zeros_like(labels)
    acc_total = 0

    for i in range(batch_size):
        pred_labels_single = pred_labels[i, :]
        labels_single = labels[i, :]

        best_acc = -1
        best_perm = None

        for j in range(permutations.shape[0]):
            permutation = permutations[j, :]
            # Apply permutation to prediction instead of label
            pred_perm = permutation[pred_labels_single.astype(int)]
            acc_under_perm = compute_accuracy_multiclass_batch(pred_perm, labels_single)

            if acc_under_perm > best_acc:
                best_acc = acc_under_perm
                best_perm = permutation

        # 最优 permutation 作用于 prediction，确保其标签顺序与 labels 一致
        best_matched_preds[i, :] = best_perm[pred_labels_single.astype(int)]
        acc_total += best_acc

    acc = acc_total / batch_size
    # acc = (acc - 1 / n_classes) / (1 - 1 / n_classes)  # Normalized Accuracy

    return acc, best_matched_preds

# #The original verion of computing the accuracy of the muti-class
# def original_compute_accuracy_multiclass(pred_llh, labels, n_classes):
#     pred_llh = pred_llh.data.cpu().numpy()
#     labels = labels.data.cpu().numpy()
#     batch_size = pred_llh.shape[0]
#     pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)
#     acc = 0
#     permutations = permuteposs(n_classes)
#
#     for i in range(batch_size):
#         pred_labels_single = pred_labels[i, :]
#         labels_single = labels[i, :]
#         for j in range(permutations.shape[0]):
#             permutation = permutations[j, :]
#             labels_under_perm = permutations[j, labels_single.astype(int)]
#
#             acc_under_perm = compute_accuracy_multiclass_batch(pred_labels_single, labels_under_perm)
#             if (j == 0):
#                 acc_single = acc_under_perm
#             else:
#                 acc_single = np.max([acc_single, acc_under_perm])
#
#         acc += acc_single
#     acc = acc / labels.shape[0]
#     acc = (acc - 1 / n_classes) / (1 - 1 / n_classes)
#
#     return acc

# def compute_accuracy_multiclass_local_refinement(pred_llh, labels, n_classes):
#     pred_llh = pred_llh.data.cpu().numpy()
#     labels = labels.data.cpu().numpy()
#     batch_size = pred_llh.shape[0]
#     pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)
#     acc = 0
#     permutations = permuteposs(n_classes)
#     max_acc_labels = []  # Store the labels corresponding to the maximum accuracy
#     adjacency_matrices = []  # Store the adjacency matrix of each sample
#
#     for i in range(batch_size):
#         pred_labels_single = pred_labels[i, :]
#         labels_single = labels[i, :]
#         max_acc_single = -1  # 记录最大 accuracy
#         best_labels = None  # 记录 accuracy 最大时的标签
#
#         for j in range(permutations.shape[0]):
#             permutation = permutations[j, :]
#             labels_under_perm = permutations[j, labels_single.astype(int)]
#
#             acc_under_perm = compute_accuracy_multiclass_batch(pred_labels_single, labels_under_perm)
#
#             if j == 0:
#                 acc_single = acc_under_perm
#                 max_acc_single = acc_under_perm
#                 best_labels = labels_under_perm  # 初次赋值
#             else:
#                 if acc_under_perm > max_acc_single:
#                     max_acc_single = acc_under_perm
#                     best_labels = labels_under_perm  # 记录 accuracy 最大时的标签
#                 acc_single = np.max([acc_single, acc_under_perm])
#
#         max_acc_labels.append(best_labels)  # 存储 accuracy 最大时的标签
#
#         # 构建邻接矩阵
#         num_nodes = len(best_labels)
#         adjacency_matrix = np.zeros((num_nodes, num_nodes))
#         for x in range(num_nodes):
#             for y in range(num_nodes):
#                 adjacency_matrix[x, y] = 1 if best_labels[x] == best_labels[y] else 0
#         adjacency_matrices.append(adjacency_matrix)  # 存储矩阵
#
#         acc += acc_single
#
#     acc = acc / labels.shape[0]
#     acc = (acc - 1 / n_classes) / (1 - 1 / n_classes)
#
#     return acc, max_acc_labels, adjacency_matrices  # 返回 accuracy、最佳标签和邻接矩阵

def permuteposs(n_classes):
    permutor = Permutor(n_classes)
    permutations = permutor.return_permutations()
    return permutations


class Permutor:
    def __init__(self, n_classes):
        self.row = 0
        self.n_classes = n_classes
        self.collection = np.zeros([math.factorial(n_classes), n_classes])

    def permute(self, arr, l, r): 
        if l==r: 
            self.collection[self.row, :] = arr
            self.row += 1
        else: 
            for i in range(l,r+1): 
                arr[l], arr[i] = arr[i], arr[l] 
                self.permute(arr, l+1, r) 
                arr[l], arr[i] = arr[i], arr[l]

    def return_permutations(self):
        self.permute(np.arange(self.n_classes), 0, self.n_classes-1)
        return self.collection
                
