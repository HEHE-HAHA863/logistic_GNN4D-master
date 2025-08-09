import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


# def compute_operators(W, J):
#     N = W.shape[0]
#     d = W.sum(1)
#     D = np.diag(d)
#     QQ = W.copy()
#     WW = np.zeros([N, N, J + 2])
#     WW[:, :, 0] = np.eye(N)
#     for j in range(J):
#         WW[:, :, j + 1] = QQ.copy()
#         QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
#     WW[:, :, J + 1] = D
#     WW = np.reshape(WW, [N, N, J + 2])
#     x = np.reshape(d, [N, 1])
#     return WW, x

# def compute_operators_local_refinement(W, J, sam_com_matrix, start_x):
#     N = W.shape[0]
#     d = W.sum(1)
#     D = np.diag(d)
#     QQ = W.copy()
#     WW = np.zeros([N, N, J + 3])
#     WW[:, :, 0] = np.eye(N)
#     for j in range(J):
#         WW[:, :, j + 1] = QQ.copy()
#         QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
#     WW[:, :, J + 1] = D
#     WW[:, :, J + 2] = sam_com_matrix
#     WW = np.reshape(WW, [N, N, J + 3])
#     x = start_x
#     return WW, x

def compute_operators_local_refinement(W, J, sam_com_matrix, start_x, n_classes):
    N = W.shape[0]                           # 节点数
    d = W.sum(1)                             # 节点度
    D = np.diag(d)                           # 度矩阵 D
    QQ = W.copy()                            # 临时变量
    WW = np.zeros([N, N, J + 3])             # 初始化特征矩阵，多出1个通道

    WW[:, :, 0] = np.eye(N)                  # 通道 0: 单位阵 I
    for j in range(J):                       # 通道 1~J: W^1, W^2, ..., W^J（带裁剪）
        WW[:, :, j + 1] = QQ.copy()
        QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))  # 元素限制为1

    WW[:, :, J + 1] = D                      # 通道 J+1: 度矩阵
    WW[:, :, J + 2] = np.ones((N, N))        # 通道 J+3: 全1矩阵

    # WW[:, :, J + 3] = sam_com_matrix         # 通道 J+2: 社区相似度矩阵

    WW = np.reshape(WW, [N, N, J + 3])
    # x = np.reshape(start_x,[N,1])                              # 返回输入 x，不做改变
    # x = np.reshape(start_x,[N,1])                              # 返回输入 x，不做改变
    labels = np.asarray(start_x).reshape(N).astype(int)  # 支持 (N,1)/(1,N)/

    x = np.eye(n_classes, dtype=np.float32)[labels]  # [N, n_classes]

    return WW, x


def get_gnn_inputs_local_refinement(W, J, sam_com_matrix, start_x, n_classes):
    sam_com_matrix = np.array(sam_com_matrix)[0]
    W = W[0, :, :]
    WW, x = compute_operators_local_refinement(W, J, sam_com_matrix, start_x, n_classes)
    WW = WW.astype(float)
    x = x.astype(float)
    WW = torch.tensor(WW, requires_grad=True).unsqueeze(0)
    x = torch.tensor(x, requires_grad=True).unsqueeze(0)
    # x = x.clone().detach().requires_grad_(True).unsqueeze(0)
    return WW, x



