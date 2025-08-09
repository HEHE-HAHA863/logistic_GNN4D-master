import glob
import csv
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import os
from data_generator import Generator
from load import get_gnn_inputs
from models import GNN_multiclass
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from get_infor_from_first_step import get_second_period_labels_single_real, \
    test_first_get_second_period_labels_single_real
from losses import compute_loss_multiclass, original_compute_accuracy_multiclass
from load_local_refinement import get_gnn_inputs_local_refinement

parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
###############################################################################

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(6000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--p_SBM', nargs='?', const=1, type=float,
                    default=0.3)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.1)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=2)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='SBM_multiclass')
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--path_local_refinement', nargs='?', const=1, type=str, default='')
parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--filename_existing_gnn_local_refinement', nargs='?', const=1, type=str, default='')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
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
                    default=20)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default=3)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=50)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=50)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()


def get_available_device():
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            torch.zeros(1).cuda()
            return torch.device(f"cuda:{i}")
        except RuntimeError:
            continue
    return torch.device("cpu")


device = get_available_device()


def safe_to_dense_adj(edge_index, num_nodes):
    if edge_index.size(1) == 0:
        W = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float)
    else:
        W = to_dense_adj(edge_index, max_num_nodes=num_nodes)
    return W


def load_graph_data(graph_path, label_path, n_classes):
    """从文件加载图数据并确保节点一致性"""
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

    all_nodes = sorted(edge_nodes.union(label_nodes))
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

    edge_index = []
    if edge_list:
        edge_index = []
        for u, v in edge_list:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    labels = torch.tensor([node_labels[node] for node in all_nodes], dtype=torch.long)
    x = torch.eye(len(all_nodes))

    data = Data(x=x, edge_index=edge_index)
    # batch = torch.zeros(data.num_nodes, dtype=torch.long)
    W = safe_to_dense_adj(data.edge_index, num_nodes=labels.size(0))

    W_np = W.squeeze(0).numpy()
    N = W_np.shape[0]
    shuffle_idx = np.random.permutation(N)
    W_np = W_np[shuffle_idx][:, shuffle_idx]
    W_np = np.expand_dims(W_np, 0)

    labels = labels.unsqueeze(0)
    labels = labels[:, shuffle_idx]

    return W_np, labels


def get_file_pairs(n_classes, graph_dir="graphs", label_dir="labels"):
    """获取匹配的图文件对"""
    graph_pattern = os.path.join(graph_dir, f"subgraph_ncls_{n_classes}_subgraph_*.csv")
    graph_files = sorted(glob.glob(graph_pattern))
    file_pairs = []

    for g_path in graph_files:
        base_name = os.path.basename(g_path)
        hash_id = base_name.split('_')[-1].split('.')[0]
        l_path = os.path.join(label_dir, f"sublabel_ncls_{n_classes}_subgraph_{hash_id}.csv")
        if os.path.exists(l_path):
            file_pairs.append((g_path, l_path))

    return file_pairs


def train_single_first_period(gnn, optimizer, n_classes, graph_path="", labels_path=""):
    W, labels = load_graph_data(graph_path, labels_path, n_classes)
    labels = labels.type(dtype_l).to(device)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1) / 2

    WW, x = get_gnn_inputs(W, args.J)
    WW = WW.to(device)
    x = x.to(device)

    pred = gnn(WW.type(dtype), x.type(dtype))
    loss = compute_loss_multiclass(pred, labels, n_classes)
    loss.backward()
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    acc = original_compute_accuracy_multiclass(pred, labels, n_classes)
    loss_value = loss.item()

    WW = None
    x = None
    torch.cuda.empty_cache()

    return loss_value, acc


def train_first_period(gnn, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn.train()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "amazon_all_comm_max500_min150")
    graph_dir = os.path.join(dataset_dir, "train_graph")
    label_dir = os.path.join(dataset_dir, "train_label")

    file_pairs = get_file_pairs(n_classes, graph_dir=graph_dir, label_dir=label_dir)

    if not file_pairs:
        raise ValueError(f"No training data found for n_classes={n_classes}")

    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    iters = len(file_pairs)
    print("iters:", iters)

    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])

    for it in range(iters):
        g_path, l_path = file_pairs[it]
        loss_single, acc_single = train_single_first_period(gnn, optimizer, n_classes, g_path, l_path)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()


def train_single_local_refinement(gnn_first_period, gnn_local_refine, n_classes, optimizer, graph_path="",
                                  label_path=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_com_matrix, true_labels, pred_labels, W = get_second_period_labels_single_real(gnn_first_period, n_classes,
                                                                                       args, graph_path, label_path)
    true_labels = true_labels.type(dtype_l)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        true_labels = (true_labels + 1) / 2

    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_labels)
    WW = WW.to(device)
    x = x.to(device)
    true_labels = true_labels.to(device)

    pred = gnn_local_refine(WW.type(dtype), x.type(dtype))
    loss = compute_loss_multiclass(pred, true_labels, n_classes)
    gnn_local_refine.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(gnn_local_refine.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc = original_compute_accuracy_multiclass(pred, true_labels, n_classes)
    loss_value = float(loss.cpu().detach().numpy())

    WW = None
    x = None
    return loss_value, acc


def train_local_refinement(gnn_first_period, gnn_local_refine, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn_local_refine.train()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "amazon_all_comm_max500_min150")
    graph_dir = os.path.join(dataset_dir, "train_graph")
    label_dir = os.path.join(dataset_dir, "train_label")

    file_pairs = get_file_pairs(n_classes, graph_dir=graph_dir, label_dir=label_dir)

    if not file_pairs:
        raise ValueError(f"No training data found for n_classes={n_classes}")

    optimizer = torch.optim.Adamax(gnn_local_refine.parameters(), lr=args.lr)

    iters = len(file_pairs)

    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])

    for it in range(iters):
        g_path, l_path = file_pairs[it]
        loss_single, acc_single = train_single_local_refinement(gnn_first_period, gnn_local_refine, n_classes,
                                                                optimizer, g_path, l_path)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()


def test_single_local_refinement_real(gnn_first_period, gnn_local_refinement, n_classes, args, graph_path, labels_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_com_matrix, true_labels, pred_single_label, W, loss_test_first, acc_test_first = test_first_get_second_period_labels_single_real(
        gnn_first_period, n_classes, args, graph_path, labels_path)
    W, labels = W, true_labels
    labels = labels.type(dtype_l)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1) / 2

    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_single_label)
    if torch.cuda.is_available():
        WW = WW.to(device)
        x = x.to(device)

    pred_single = gnn_local_refinement(WW.type(dtype), x.type(dtype))
    loss_test_second = compute_loss_multiclass(pred_single, labels, n_classes)
    acc_test_second = original_compute_accuracy_multiclass(pred_single, labels, n_classes)

    if torch.cuda.is_available():
        loss_test_second = float(loss_test_second.data.cpu().numpy())
    else:
        loss_test_second = float(loss_test_second.data.numpy())

    WW = None
    x = None
    return loss_test_first, acc_test_first, acc_test_second, loss_test_second


def test_local_refinement(gnn_first_period, gnn_local_refinement, n_classes, iters=args.num_examples_test,
                          filename="test_results_real.csv"):
    gnn_first_period.train()
    gnn_local_refinement.train()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "amazon_all_comm_max500_min150")
    graph_dir = os.path.join(dataset_dir, "test_graph")
    label_dir = os.path.join(dataset_dir, "test_label")

    file_pairs = get_file_pairs(n_classes, graph_dir=graph_dir, label_dir=label_dir)

    if not file_pairs:
        raise ValueError(f"No training data found for n_classes={n_classes}")

    iters = len(file_pairs)

    loss_lst_first = np.zeros([iters])
    acc_lst_first = np.zeros([iters])
    loss_lst_second = np.zeros([iters])
    acc_lst_second = np.zeros([iters])

    for it in range(iters):
        g_path, l_path = file_pairs[it]
        loss_test_first, acc_test_first, acc_test_second, loss_test_second = test_single_local_refinement_real(
            gnn_first_period, gnn_local_refinement, n_classes, args, g_path, l_path)
        loss_lst_first[it] = loss_test_first
        acc_lst_first[it] = acc_test_first
        loss_lst_second[it] = loss_test_second
        acc_lst_second[it] = acc_test_second
        torch.cuda.empty_cache()

    first_avg_test_acc = np.mean(acc_lst_first)
    first_std_test_acc = np.std(acc_lst_first)
    second_avg_test_acc = np.mean(acc_lst_second)
    second_std_test_acc = np.std(acc_lst_second)

    df = pd.DataFrame([{
        "n_classes": args.n_classes,
        "J": args.J,
        "first_avg_test_acc": first_avg_test_acc,
        "first_std_test_acc": first_std_test_acc,
        "second_avg_test_acc": second_avg_test_acc,
        "second_std_test_acc": second_std_test_acc
    }])

    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    gen = Generator()
    gen.N_train = args.N_train
    args.N_test = args.N_train
    gen.N_test = args.N_test

    gen.edge_density = args.edge_density
    gen.p_SBM = args.p_SBM
    gen.q_SBM = args.q_SBM
    gen.random_noise = args.random_noise
    gen.noise = args.noise
    gen.noise_model = args.noise_model
    gen.generative_model = args.generative_model
    gen.n_classes = args.n_classes

    torch.backends.cudnn.enabled = False

    filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(
        args.num_examples_train)
    path_plus_name = os.path.join(args.path_gnn, filename)

    if (args.generative_model == 'SBM_multiclass'):
        gnn_first_period = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)

    if torch.cuda.is_available():
        gnn_first_period = gnn_first_period.to(device)

    if (args.generative_model == 'SBM_multiclass'):
        train_first_period(gnn_first_period, args.n_classes)

    filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(
        args.num_examples_train) + '_local_refinement'
    path_plus_name = os.path.join(args.path_gnn, filename)

    if (args.generative_model == 'SBM_multiclass'):
        gnn_local_refinement = GNN_multiclass(args.num_features, args.num_layers, args.J + 3, n_classes=args.n_classes)

    if torch.cuda.is_available():
        gnn_local_refinement = gnn_local_refinement.to(device)

    train_local_refinement(gnn_first_period, gnn_local_refinement, n_classes=args.n_classes,
                           iters=args.num_examples_train)
    test_local_refinement(gnn_first_period, gnn_local_refinement, args.n_classes, iters=args.num_examples_test)