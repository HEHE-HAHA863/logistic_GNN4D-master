import numpy as np
import os
from data_generator import Generator
from load import get_gnn_inputs
from models import GNN_multiclass
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from main_gnn import train, train_single, test, test_single

from get_infor_from_first_step import get_second_period_labels_single,test_get_second_period_labels
from losses import compute_loss_multiclass, original_compute_accuracy_multiclass
from load_local_refinement import get_gnn_inputs_local_refinement

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
                    default=0.1)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.05)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=2)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
#########################
#parser.add_argument('--generative_model', nargs='?', const=1, type=str,
#                    default='ErdosRenyi')
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='SBM_multiclass')
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default=default_path)
parser.add_argument('--path_local_refinement', nargs='?', const=1, type=str, default=default_path)

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
                    default=50)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default= 2)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

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


def train_single_local_refinement(gnn, optimizer, n_classes, it, W, pred_labels, true_labels, sam_com_matrix):
    start = time.time()
    W = W
    true_labels = true_labels.type(dtype_l)

    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        true_labels = (true_labels + 1)/2

    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_labels)

    if (torch.cuda.is_available()):
        WW.cuda()
        x.cuda()

    pred = gnn(WW.type(dtype), x.type(dtype))

    loss = compute_loss_multiclass(pred, true_labels, n_classes)
    gnn.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc = original_compute_accuracy_multiclass(pred, true_labels, n_classes)

    elapsed = time.time() - start

    if(torch.cuda.is_available()):
        loss_value = float(loss.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())

    # info = ['iter', 'avg loss', 'avg acc', 'edge_density',
    #         'noise', 'model', 'elapsed']
    # out = [it, loss_value, acc, args.edge_density,
    #        args.noise, 'GNN', elapsed]
    # print(template1.format(*info))
    # print(template2.format(*out))

    del WW
    del x

    return loss_value, acc


def train_local_refinement(gnn,all_sam_com_matrix,all_true_labels,
                           all_pred_labels, all_W, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        sam_com_matrix = np.array(all_sam_com_matrix[it])
        true_labels = all_true_labels[it]
        pred_labels = all_pred_labels[it].T #We have to transpose because the predicted_labels do not match the input of the GMLU
        # pred_labels = pred_labels.squeeze(0)
        W = all_W[it]
        loss_single, acc_single = train_single_local_refinement(gnn, optimizer, n_classes, it,
                                                                W, pred_labels, true_labels, sam_com_matrix)

        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()
    print ('Avg train loss', np.mean(loss_lst))
    print ('Avg train acc', np.mean(acc_lst))
    print ('Std train acc', np.std(acc_lst))


def test_single_local_refinement(gnn, n_classes, it, sam_com_matrix, true_labels, pred_single_labels, W):
    start = time.time()

    # --- 1. 原始数据预处理 ---
    W_np = W.squeeze(0).cpu().numpy() if isinstance(W, torch.Tensor) else W
    labels = true_labels.type(torch.long)
    if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
        labels = (labels + 1) // 2  # 保持原二分类标签处理

    # --- 2. 原始特征计算（邻接矩阵和拉普拉斯矩阵特征）---
    W_sym = (W_np + W_np.T) / 2
    eigvals_W, eigvecs_W = np.linalg.eigh(W_sym)
    adjacency_eigvecs = eigvecs_W[:, np.argsort(eigvals_W)[-2:]]

    D = np.diag(np.sum(W_np, axis=1))
    L = D - W_np
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    laplacian_eigvecs = eigvecs_L[:, np.argsort(eigvals_L)[:2]]

    # --- 3. 新增：计算 sign(W * pred_single_labels) 用于比较 ---
    pred_labels_np = pred_single_labels.squeeze(0).cpu().numpy()
    # 二分类转换为{-1,1}，多分类保持原样
    binary_pred = np.sign(pred_labels_np * 2 - 1) if n_classes == 2 else pred_labels_np
    W_times_pred = W_np @ binary_pred  # 矩阵乘法计算W*pred
    sign_W_times_pred = np.sign(W_times_pred)  # 取符号

    # --- 4. GNN前向传播（原始功能）---
    WW, x = get_gnn_inputs_local_refinement(W, args.J, sam_com_matrix, pred_single_labels)
    if torch.cuda.is_available():
        WW, x = WW.cuda(), x.cuda()

    pred = gnn(WW.float(), x.float())
    penultimate_features = gnn.get_penultimate_output().detach().cpu().numpy().squeeze(0)

    # --- 5. 综合保存所有结果到Excel（原始+新增）---
    output_filename = "all_results_with_comparison.xlsx"
    df = pd.DataFrame({
        # 原始输出
        'GNN_Feature1': penultimate_features[:, 0],
        'GNN_Feature2': penultimate_features[:, 1],
        'Adj_EigVec1': adjacency_eigvecs[:, 0],
        'Adj_EigVec2': adjacency_eigvecs[:, 1],
        'Lap_EigVec1': laplacian_eigvecs[:, 0],
        'Lap_EigVec2': laplacian_eigvecs[:, 1],
        'Labels': labels.squeeze(0).cpu().numpy(),
        # 新增比较字段
        'Pred_Labels': pred_labels_np,  # 模型预测标签
        'W_times_Pred': W_times_pred,  # W矩阵乘以预测标签
        'Sign_W_times_Pred': sign_W_times_pred  # 符号化结果
    })

    # 保存逻辑（保留原始方式）
    if it == 0:
        df.to_excel(output_filename, sheet_name=f'Iter_{it}', index=False)
    else:
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=f'Iter_{it}', index=False)

    # --- 6. 原始损失和准确率计算 ---
    loss = compute_loss_multiclass(pred, labels, n_classes)
    acc = original_compute_accuracy_multiclass(pred, labels, n_classes)

    # --- 7. 原始日志打印 ---
    print(f"iter {it}: loss={loss.item():.4f}, acc={acc:.2f}%, time={time.time() - start:.2f}s")

    return loss.item(), acc

def test_local_refinement(gnn, sam_com_matrices, true_labels_lst, pred_labels_lst, W_lst, n_classes, iters=args.num_examples_test):
    gnn.train()
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])

    for it in range(iters):
        # 依次取出列表中的元素
        sam_com_matrix = sam_com_matrices[it]
        true_labels = true_labels_lst[it]
        pred_single_label = pred_labels_lst[it].T
        W = W_lst[it]

        loss_single, acc_single = test_single_local_refinement(gnn, n_classes, it, sam_com_matrix, true_labels, pred_single_label,W)

        loss_lst[it] = loss_single
        acc_lst[it] = acc_single

        torch.cuda.empty_cache()

    print('Avg test loss', np.mean(loss_lst))
    print('Avg test acc', np.mean(acc_lst))
    print('Std test acc', np.std(acc_lst))

    return loss_lst, acc_lst

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    gen = Generator()
    gen.N_train = args.N_train
    gen.N_test = args.N_test
    gen.edge_density = args.edge_density
    gen.p_SBM = args.p_SBM
    gen.q_SBM = args.q_SBM
    gen.random_noise = args.random_noise
    gen.noise = args.noise
    gen.noise_model = args.noise_model
    gen.generative_model = args.generative_model
    gen.n_classes = args.n_classes


    torch.backends.cudnn.enabled=False




########Train and get the first period GNN #######################################
    # if (args.mode == 'test'):
    #     print ('In testing mode')
    #     filename = args.filename_existing_gnn
    #     path_plus_name = os.path.join(args.path_gnn, filename)
    #     if ((filename != '') and (os.path.exists(path_plus_name))):
    #         print ('Loading gnn ' + filename)
    #         gnn = torch.load(path_plus_name)
    #         if torch.cuda.is_available():
    #             gnn.cuda()
    #     else:
    #         print ('No such a gnn exists; creating a brand new one')
    #         if (args.generative_model == 'SBM_multiclass'):
    #             gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
    #         filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
    #         path_plus_name = os.path.join(args.path_gnn, filename)
    #         if torch.cuda.is_available():
    #             gnn.cuda()
    #         print ('Training begins')


    if (args.mode == 'train'):
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            filename = filename + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)

        else:
            print ('No such a gnn exists; creating a brand new one')
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)

        print ('total num of params:', count_parameters(gnn))

        if torch.cuda.is_available():
            gnn.cuda()
        print ('Training begins')
        if (args.generative_model == 'SBM_multiclass'):
            print('The training result of the first period')
            train(gnn, gen, args.n_classes)
        print ('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn, path_plus_name)

        # args.filename_existing_gnn = "gnn_J4_lyr20_Ntr50_num6000"

#################################################################################################
    # Train and test the local refinement
    if (args.mode == 'test'):
        print ('In testing mode')
        filename = args.filename_existing_gnn
        filename_local_refinement = args.filename_existing_gnn_local_refinement

        path_plus_name = os.path.join(args.path_gnn, filename) #The first GNN trained from the first period
        path_local_refinement = os.path.join(args.path_local_refinement, filename_local_refinement)
        #The second GNN used to do the local refinement

        if ((filename != '') and (filename_local_refinement != '') and (os.path.exists(path_plus_name)) & (os.path.exists(path_local_refinement))):
            print ('Loading gnn_first_period ' + filename)
            print ('Loading local refinement ' + path_local_refinement)

            gnn_first_period = torch.load(path_plus_name)
            gnn_local_refinement = torch.load(path_local_refinement)

            if torch.cuda.is_available():
                gnn_first_period.cuda()
                gnn_local_refinement.cuda()
        else:
            print ('No such a gnn exists; creating a brand new one')
            if (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Training begins')


    elif (args.mode == 'train'):
        filename = args.filename_existing_gnn_local_refinement
        path_plus_name = os.path.join(args.path_gnn, filename)

        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn_first_period = torch.load(path_plus_name)
            filename = filename + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_local_refinement'
            path_plus_name = os.path.join(args.path_gnn, filename)
        else:
            print ('No such a gnn exists; creating a brand new one')
            gnn_first_period = torch.load('gnn_J4_lyr20_Ntr50_num6000')
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_local_refinement'
            path_plus_name = os.path.join(args.path_gnn, filename)
            if (args.generative_model == 'SBM_multiclass'):
                gnn_local_refinement = GNN_multiclass(args.num_features, args.num_layers, args.J + 3, n_classes=args.n_classes)

        print ('total num of params:', count_parameters(gnn_local_refinement))

        if torch.cuda.is_available():
            gnn_local_refinement.cuda()
        print ('Training begins')
        if (args.generative_model == 'SBM_multiclass'):
            all_sam_com_matrices, all_true_labels, all_pred, all_W = get_second_period_labels_single(gnn_first_period, gen,
                                                                                              args.n_classes, args.num_examples_train, args)
            print('Second period labels are already prepared')
            print("The result of the training local refinement is")
            train_local_refinement(gnn_local_refinement, all_sam_com_matrices, all_true_labels, all_pred, all_W)
        print ('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn_local_refinement.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn_local_refinement, path_plus_name)


    # print ('Testing the GNN:')
    # if args.eval_vs_train:
    #     print ('model status: eval')
    #     gnn.eval()
    # else:
    #     print ('model status: train')
    #     gnn.train()
    #
    print("The result of the test in the first period")
    sam_com_matrices, true_labels_lst, pred_labels_lst, W_lst = test_get_second_period_labels(gnn_first_period, gen, args.n_classes,args.num_examples_test, args)
    print("The result of the test with the local refinement")
    test_local_refinement(gnn_local_refinement, sam_com_matrices, true_labels_lst, pred_labels_lst, W_lst, args.n_classes, iters = args.num_examples_test)
    #
    # print ('total num of params:', count_parameters(gnn))


