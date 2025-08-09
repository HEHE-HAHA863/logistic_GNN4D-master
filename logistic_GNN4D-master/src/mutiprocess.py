import os
import sys
import subprocess
from itertools import product

# 1. 获取当前PyCharm使用的Python解释器路径
PYTHON_PATH = sys.executable
print(f"当前Python解释器路径: {PYTHON_PATH}")

# # 构造合法配对的参数组合（添加 num_layers = 30, 50, 100）
# param_config = []
#
# valid_settings = [
#     (3, 0.1, 0.05),
#     (3, 0.2, 0.1),
#     (4, 0.1, 0.05),
#     (4, 0.2, 0.1),
# ]
#
# num_layers_list = [30, 50, 100]
#
# for n_classes, p, q in valid_settings:
#     for num_layers in num_layers_list:
#         param_config.append({
#             'n_classes': n_classes,
#             'p_SBM': p,
#             'q_SBM': q,
#             'num_layers': num_layers,
#         })
#
#
#
# # 3. 固定参数
# fixed_params = {
#     'num_examples_train': 6000,
#     'num_examples_test': 20,
#     'num_features': 8
# }
import numpy as np

# 参数初始化
N = 1000
logN_div_N = np.log(N) / N  # ≈ 0.0069

c1_values = [2, 5, 10]           # 控制稀疏程度
n_classes_list = [2, 3, 4]          # 社区数
fixed_num_layers = 50            # 固定 GNN 层数

param_config = []

for c1 in c1_values:
    p = round(c1 * logN_div_N, 4)
    q = round(0.5 * p, 4)  # 默认 c2 = 0.5 * c1
    for n_classes in n_classes_list:
        param_config.append({
            'n_classes': n_classes,
            'p_SBM': p,
            'q_SBM': q,
            'num_layers': fixed_num_layers,
            'c1': c1,
            'c2': 0.5 * c1,
        })

# 固定参数
fixed_params = {
    'N_train': N,
    'N_test': N,
    'num_examples_train': 6000,
    'num_examples_test': 20,
    'num_features': 8
}

# 4. 运行所有实验
for idx, config in enumerate(param_config, start=1):
    nc = config['n_classes']
    p = config['p_SBM']
    q = config['q_SBM']
    num_layers = config['num_layers']
    job_name = f"train_gnn_{idx}_nc{nc}_p{p}_q{q}_nlayers{num_layers}"
    print(f"\n▶ 开始任务 [{idx}]: {job_name}")
    # 后续可以加入运行函数，例如 run_exp(config | fixed_params)

    cmd = [
        PYTHON_PATH,
        "main_gnn_local_refi_final.py",
        "--path_gnn", "",
        "--filename_existing_gnn", "",
        "--num_examples_train", str(fixed_params['num_examples_train']),
        "--num_examples_test", str(fixed_params['num_examples_test']),
        "--p_SBM", str(p),
        "--q_SBM", str(q),
        "--generative_model", "SBM_multiclass",
        "--batch_size", "1",
        "--mode", "train",
        "--J", "4",
        "--clip_grad_norm", "40.0",
        "--num_features", str(fixed_params['num_features']),
        "--num_layers", str(num_layers),
        "--J", "4",
        "--N_train", "1000",
        "--N_test", "1000",
        "--print_freq", "1",
        "--n_classes", str(nc),
        "--lr", "0.004"
    ]

    # 打印执行的完整命令（调试用）
    print("执行命令:", " ".join(cmd))

    # 运行并捕获输出
    out_file = f"{job_name}.out"
    err_file = f"{job_name}.err"
    with open(out_file, "w") as out, open(err_file, "w") as err:
        try:
            subprocess.run(cmd, stdout=out, stderr=err, check=True)
            print(f"✅ 任务 {idx} 完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 任务 {idx} 失败! 返回码: {e.returncode}")
            with open(err_file, "a") as f:
                f.write(f"\nProcess failed with return code {e.returncode}")

# 5. 最终验证
print("\n所有任务提交完成，请检查 .out 和 .err 文件")
