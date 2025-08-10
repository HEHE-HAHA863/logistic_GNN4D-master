import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


def GMul(W, x):
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-3]
    J = W_size[-1]
    W_lst = W.split(1, 3)
    if N > 5000:
        output_lst = []
        for W in W_lst:
            output_lst.append(torch.bmm(W.squeeze(3),x))
        output = torch.cat(output_lst, 1)
    else:
        W = torch.cat(W_lst, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
        output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class gnn_atomic(nn.Module):
    def __init__(self, feature_maps, J):
        super(gnn_atomic, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, WW, x):
        x = GMul(WW, x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x1 = F.relu(self.fc1(x)) # has size (bs*N, num_outputs)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = self.bn(x)
        x = x.view(*x_size[:-1], self.num_outputs)
        return WW, x

# class gnn_atomic(nn.Module):
#     def __init__(self, feature_maps, J):
#         super().__init__()
#         self.num_inputs = J * feature_maps[0]
#         self.num_outputs = feature_maps[2]
#         self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
#         self.bn  = nn.BatchNorm1d(self.num_outputs)
#
#     def forward(self, WW, x):
#         x = GMul(WW, x)                      # (bs, N, Cin)
#         bs, N, _ = x.shape
#         x = x.contiguous().view(-1, self.num_inputs)     # (bs*N, Cin)
#         h = self.fc1(x)                                   # (bs*N, Cout/2)
#         x = torch.cat([F.relu(h), h], dim=1)              # (bs*N, Cout)
#         x = self.bn(x)                                    # BN on (N, C)
#         x = x.view(bs, N, self.num_outputs)
#         return WW, x

    
class gnn_atomic_final(nn.Module):
    def __init__(self, feature_maps, J, n_classes):
        super(gnn_atomic_final, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = n_classes
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, WW, x):
        x = GMul(WW, x) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1)
        x = self.fc(x) # has size (bs*N, num_outputs)
        x = x.view(*x_size[:-1], self.num_outputs)
        return WW, x

class GNN_multiclass(nn.Module):
    def __init__(self, num_features, num_layers, J, n_classes=2):
        super(GNN_multiclass, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers

        # 修改1：将倒数第二层的输出维度设为2
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        # 修改2：单独设置倒数第二层的输出维度为n_classes,以此来观察是不是两个特征向量
        self.featuremap_penultimate = [num_features, num_features, 2 * n_classes]  # 输出n_classes维特征
        self.featuremap_end = [2 * n_classes, n_classes, n_classes]  # 输入维度改为n_classes

        self.layer0 = gnn_atomic(self.featuremap_in, J)

        # 修改3：前 num_layers-1 层使用原始维度
        for i in range(num_layers - 1):
            module = gnn_atomic(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)

        # 修改4：倒数第二层输出2维特征
        self.penultimate_layer = gnn_atomic(self.featuremap_penultimate, J)
        # 修改5：最后一层输入维度匹配为2
        self.layerlast = gnn_atomic_final(self.featuremap_end, J, n_classes)

        # 存储倒数第二层输出的钩子
        self.penultimate_output = None

    def forward(self, W, x):
        cur = self.layer0(W, x)

        # 前 num_layers-1 层
        for i in range(self.num_layers - 1):
            cur = self._modules['layer{}'.format(i + 1)](*cur)

        # 获取倒数第二层输出
        penultimate_out = self.penultimate_layer(*cur)
        self.penultimate_output = penultimate_out[1]  # 存储特征矩阵

        # 最后一层
        out = self.layerlast(*penultimate_out)
        return out[1]

    # 新增方法：获取倒数第二层输出
    def get_penultimate_output(self):
        return self.penultimate_output

class GNN_multiclass_second_period(nn.Module):
    def __init__(self, num_features, num_layers, J, n_classes=2):
        super().__init__()  # ✅ 正确调用父类初始化
        self.num_features = num_features
        self.num_layers = num_layers

        # 修改1：将倒数第二层的输出维度设为n_classes
        self.featuremap_in = [n_classes, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        # 修改2：单独设置倒数第二层的输出维度为n_classes
        self.featuremap_penultimate = [num_features, num_features, num_features]  # 输出2维特征
        self.featuremap_end = [num_features, n_classes, n_classes]  # 输入维度改为2

        self.layer0 = gnn_atomic(self.featuremap_in, J)

        # 修改3：前 num_layers-1 层使用原始维度
        for i in range(num_layers - 1):
            module = gnn_atomic(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)

        # 修改4：倒数第二层输出n_classes维特征
        self.penultimate_layer = gnn_atomic(self.featuremap_penultimate, J)
        # 修改5：最后一层输入维度匹配为n_classes
        self.layerlast = gnn_atomic_final(self.featuremap_end, J, n_classes)

        # 存储倒数第二层输出的钩子
        self.penultimate_output = None

    def forward(self, W, x):
        cur = self.layer0(W, x)

        # 前 num_layers-1 层
        for i in range(self.num_layers - 1):
            cur = self._modules['layer{}'.format(i + 1)](*cur)

        # 获取倒数第二层输出
        penultimate_out = self.penultimate_layer(*cur)
        self.penultimate_output = penultimate_out[1]  # 存储特征矩阵

        # 最后一层
        out = self.layerlast(*penultimate_out)
        return out[1]

    # 新增方法：获取倒数第二层输出
    def get_penultimate_output(self):
        return self.penultimate_output
