import numpy as np
import os
from data_generator import Generator
from load import get_gnn_inputs
from models import GNN_multiclass
import time
import argparse
import glob
import csv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from losses import compute_loss_multiclass, compute_accuracy_multiclass
from losses import from_scores_to_labels_multiclass_batch


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

#In this function,we transform the file into the graphs and the labels we need
def safe_to_dense_adj(edge_index, num_nodes):
    if edge_index.size(1) == 0:
        # 如果是空图，返回一个全0的邻接矩阵（无连接）
        W = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float)
    else:
        W = to_dense_adj(edge_index, max_num_nodes=num_nodes)
    return W

def load_graph_data(graph_path, label_path, n_classes):
    """从文件加载图数据并确保节点一致性"""
    # 加载边列表并收集所有节点
    edge_nodes = set()
    edge_list = []
    with open(graph_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            edge_list.append([u, v])
            edge_nodes.add(u)
            edge_nodes.add(v)

    # 加载标签并收集所有节点
    label_nodes = set()
    node_labels = {}
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[0])
            label = int(row[1])
            node_labels[node_id] = label
            label_nodes.add(node_id)

    # 合并所有节点（边和标签的并集）
    all_nodes = sorted(edge_nodes.union(label_nodes))

    # 创建节点到索引的映射
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    # 转换边列表为索引格式
    edge_index = []
    if edge_list:
        edge_index = []
        for u, v in edge_list:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 生成标签张量
    labels = torch.tensor([node_labels[node] for node in all_nodes], dtype=torch.long)

    # 生成节点特征（单位矩阵）
    x = torch.eye(len(all_nodes))

    # 构建邻接矩阵
    data = Data(x=x, edge_index=edge_index, y=labels.unsqueeze(0))
    W = safe_to_dense_adj(data.edge_index, num_nodes=labels.size(0))

    # 转换为NumPy并调整维度
    W_np = W.squeeze(0).numpy()  # (N, N)

    # 创建随机排列索引
    N = W_np.shape[0]
    shuffle_idx = np.random.permutation(N)

    # 打乱邻接矩阵
    W_np = W_np[shuffle_idx][:, shuffle_idx]
    W_np = np.expand_dims(W_np, 0)  # (1, N, N)

    # 打乱标签
    labels = labels.unsqueeze(0)
    labels = labels[:,shuffle_idx]

    return W_np, labels

# return the adjacency matrix we need to do the local refinement
def get_matrix_local_refinement(pred_llh, n_classes):
    pred_llh = pred_llh.data.cpu().numpy()
    batch_size = pred_llh.shape[0]
    pred_labels = from_scores_to_labels_multiclass_batch(pred_llh)

    adjacency_matrices = []  # Store the adjacency matrix of each sample

    for i in range(batch_size):
        pred_labels_single = pred_labels[i, :]

        # Construct adjacency matrix
        num_nodes = len(pred_labels_single)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for x in range(num_nodes):
            for y in range(num_nodes):
                adjacency_matrix[x, y] = 1 if pred_labels_single[x] == pred_labels_single[y] else 0
        adjacency_matrices.append(adjacency_matrix)  # 存储矩阵

    return adjacency_matrices  # return the adjacency matrix we need to do the local refinement
    #In other words, the nodes that belong to the same community is equal to 1 , 0 otherwise

#Get the information we need to do train the local refinement
def get_second_period_labels_single(gnn, W, true_lables,n_classes, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # W, true_labels,eigvecs_top = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())

    WW, x = get_gnn_inputs(W, args.J)

    if (torch.cuda.is_available()):
        WW = WW.to(device)
        x = x.to(device)

    #Use the well-trained GNN to to predict the labels
    pred_single = gnn(WW.type(dtype), x.type(dtype))
    # true_labels = labels

    WW = None
    x = None

    adjacency_matrices = get_matrix_local_refinement(pred_single, n_classes)
    in_sample_acc, pred_single_label = compute_accuracy_multiclass(pred_single, true_lables, n_classes)

    #Here the pre_single_label is (1,50),Not (50,1)
    pred_single_label = pred_single_label.T
    # Now the pre_single_label is (50,1)

    return adjacency_matrices, pred_single_label

#In this function ,we use the real world data to get the labels ahd the graphs we need to
#do the local refinement
def get_second_period_labels_single_real(gnn,n_classes, args, graph_path= "", labels_path= ""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # W, labels = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())
    W, true_labels = load_graph_data(graph_path, labels_path, n_classes)

    labels = true_labels.type(dtype_l)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1)/2

    WW, x = get_gnn_inputs(W, args.J)

    if (torch.cuda.is_available()):
        WW = WW.to(device)
        x = x.to(device)

    #Use the well-trained GNN to to predict the labels
    pred_single = gnn(WW.type(dtype), x.type(dtype))
    # true_labels = labels

    adjacency_matrices = get_matrix_local_refinement(pred_single, n_classes)
    pred_single = pred_single.data.cpu().numpy()
    pred_single_label = from_scores_to_labels_multiclass_batch(pred_single)
    #Here the pre_single_label is (1,50),Not (50,1)
    pred_single_label = pred_single_label.T
    # Now the pre_single_label is (50,1)

    return adjacency_matrices, true_labels, pred_single_label, W


##############################
#Use the first period GNN to predict the test and get the information we need to do the local refinement
def test_first_get_second_period_labels_single(gnn, gen, n_classes, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    W, true_labels,eigvecs_top = gen.sample_otf_single(is_training=False, cuda=torch.cuda.is_available())

    WW, x = get_gnn_inputs(W, args.first_J)

    if (torch.cuda.is_available()):
        WW = WW.to(device)
        x = x.to(device)

    #Use the well-trained GNN to to predict the labels
    pred_single = gnn(WW.type(dtype), x.type(dtype))

    #Calculate the loss_test and the acc_test for the first period
    loss_test = compute_loss_multiclass(pred_single, true_labels, n_classes)
    acc_test, best_matched_pred = compute_accuracy_multiclass(pred_single, true_labels, n_classes)


    # true_labels = labels

    sam_com_matrix = get_matrix_local_refinement(pred_single, n_classes)
    # pred_single = pred_single.data.cpu().numpy()
    # pred_single_label = from_scores_to_labels_multiclass_batch(pred_single)
    # #Here the pre_single_label is (1,50),Not (50,1)
    # pred_single_label = pred_single_label.T
    WW = None
    x = None

    return sam_com_matrix, true_labels, best_matched_pred, W ,loss_test, acc_test, eigvecs_top

def imbalanced_test_first_get_second_period_labels_single(gnn, gen, n_classes, args, class_sizes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    W, true_labels,eigvecs_top = gen.imbalanced_sample_otf_single(class_sizes , is_training=False, cuda=torch.cuda.is_available())

    WW, x = get_gnn_inputs(W, args.first_J)

    if (torch.cuda.is_available()):
        WW = WW.to(device)
        x = x.to(device)

    #Use the well-trained GNN to to predict the labels
    pred_single = gnn(WW.type(dtype), x.type(dtype))

    #Calculate the loss_test and the acc_test for the first period
    loss_test = compute_loss_multiclass(pred_single, true_labels, n_classes)
    acc_test, best_matched_pred = compute_accuracy_multiclass(pred_single, true_labels, n_classes)

    sam_com_matrix = get_matrix_local_refinement(pred_single, n_classes)

    WW = None
    x = None

    return sam_com_matrix, true_labels, best_matched_pred, W ,loss_test, acc_test, eigvecs_top

def in_sample_test_first_get_second_period_labels_single(gnn, W, true_labels,n_classes, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    W, true_labels = W, true_labels

    WW, x = get_gnn_inputs(W, args.J)

    if (torch.cuda.is_available()):
        WW = WW.to(device)
        x = x.to(device)

    pred_single = gnn(WW.type(dtype), x.type(dtype))

    in_sample_loss_test = compute_loss_multiclass(pred_single, true_labels, n_classes)
    in_sample_acc_test, best_matched_pred = compute_accuracy_multiclass(pred_single, true_labels, n_classes)

    sam_com_matrix = get_matrix_local_refinement(pred_single, n_classes)

    WW = None
    x = None

    return sam_com_matrix, best_matched_pred ,in_sample_loss_test, in_sample_acc_test


def test_first_get_second_period_labels_single_real(gnn, n_classes, args, graph_path, labels_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    W, labels = load_graph_data(graph_path, labels_path, n_classes)

    # W, labels = gen.sample_otf_single(is_training=False, cuda=torch.cuda.is_available())
    labels = labels.type(dtype_l)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1)/2

    WW, x = get_gnn_inputs(W, args.J)

    if (torch.cuda.is_available()):
        WW = WW.to(device)
        x = x.to(device)

    #Use the well-trained GNN to to predict the labels
    pred_single = gnn(WW.type(dtype), x.type(dtype))

    #Calculate the loss_test and the acc_test for the first period
    loss_test = compute_loss_multiclass(pred_single, labels, n_classes)
    acc_test = compute_accuracy_multiclass(pred_single, labels, n_classes)

    true_labels = labels

    sam_com_matrix = get_matrix_local_refinement(pred_single, n_classes)
    pred_single = pred_single.data.cpu().numpy()
    pred_single_label = from_scores_to_labels_multiclass_batch(pred_single)
    #Here the pre_single_label is (1,50),Not (50,1)
    pred_single_label = pred_single_label.T

    return sam_com_matrix, true_labels, pred_single_label, W ,loss_test, acc_test

