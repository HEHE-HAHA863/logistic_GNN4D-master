import os
from multiprocessing import Pool

import multiprocessing
from torch.utils.data import Dataset
import openpyxl
from data_generator import Generator
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
from scipy.sparse import issparse

template_header = '{:<6} {:<10} {:<10} {:<10}'
template_row = '{:<6d} {:<10.4f} {:<10.2f} {:<10.2f}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cached_graphs = []
cached_labels = []

##Define the train function we need to train the first-period GNN function
def train_batch_first_period(gnn, optimizer, batch, n_classes, iter, device, args):
    """
    使用 batched 输入训练 GNN，适配用户自定义的 permutation-aware 损失函数。
    """
    Ws = batch['adj'].to(device)           # shape: (B, N, N)
    labels = batch['labels'].to(device)  # shape: (B, N)

    start = time.time()

    # ✅ 调用 batched GNN 输入处理
    WW, x = get_gnn_inputs(Ws.cpu().numpy(), args.J)  # 输出：WW: (B, N, N, J+3), x: (B, N, d)
    WW = WW.clone().detach().to(torch.float32).to(device)
    x = x.clone().detach().to(torch.float32).to(device)

    optimizer.zero_grad(set_to_none=True)

    # ✅ 前向传播，输出 shape: (B, N, n_classes)
    pred = gnn(WW, x)

    # ✅ 使用你自己的 permutation-aware loss（已内部处理 batch）
    loss = compute_loss_multiclass(pred, labels, n_classes)  # 无需 reshape
    loss.backward()

    # ✅ 梯度裁剪 + 参数更新
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()

    # ✅ 使用你自己的 accuracy 函数
    acc, _ = compute_accuracy_multiclass(pred, labels, n_classes)

    elapsed_time = time.time() - start
    loss_value = loss.item()

    # ✅ 打印信息
    print(template_header.format(*['iter', 'avg loss', 'avg acc', 'elapsed']))
    print(template_row.format(iter, loss_value, acc, elapsed_time))

    return loss_value, acc


def evaluate_on_loader(gnn, val_loader, n_classes, args ,device):
    gnn.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            Ws = batch['adj'].to(device)
            labels = batch['labels'].to(device)

            WW, x = get_gnn_inputs(Ws.cpu().numpy(), args.J)
            WW = WW.clone().detach().to(torch.float32).to(device)
            x = x.clone().detach().to(torch.float32).to(device)

            pred = gnn(WW, x)
            loss = compute_loss_multiclass(pred, labels, n_classes)
            acc, _ = compute_accuracy_multiclass(pred, labels, n_classes)

            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    return avg_loss, avg_acc


def train_first_period_with_early_stopping(
    gnn, train_loader, val_loader, n_classes,args,
    epochs=100, patience=6, save_path='best_model.pt'
):
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)

    loss_lst = []
    acc_lst = []
    val_acc_best = -1
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        gnn.train()

        for iter_idx, batch in enumerate(tqdm(train_loader)):
            loss, acc = train_batch_first_period(
                gnn=gnn,
                optimizer=optimizer,
                batch=batch,
                n_classes=n_classes,
                iter=iter_idx,
                device= device,
                args= args
            )

            loss_lst.append(loss)
            acc_lst.append(acc)
            torch.cuda.empty_cache()

        # 🧪 验证集评估
        val_loss, val_acc = evaluate_on_loader(gnn, val_loader, n_classes, args ,device= device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # ✅ 判断是否更新最优模型
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            patience_counter = 0
            torch.save(gnn.cpu(), save_path)
            print("New best model saved.")
            if torch.cuda.is_available():
                gnn = gnn.to('cuda')
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # ⛔ 提前停止
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return loss_lst, acc_lst
