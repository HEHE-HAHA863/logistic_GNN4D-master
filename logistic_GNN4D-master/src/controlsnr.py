import os
import sys
import subprocess
from itertools import product
from scipy.optimize import minimize_scalar
# def snr_objective(a, s, k, total_ab):
#     b = (total_ab - a) / (k - 1)
#     if a <= 0 or b <= 0:  # a, b 都必须为正
#         return 1e9
#     snr = ((a - b)**2) / (k * (a + (k - 1) * b))
#     return abs(snr - s)
#
# def find_a_given_snr(s, k, total_ab):
#     res = minimize_scalar(snr_objective, args=(s, k, total_ab), bounds=(1e-2, total_ab - 1e-2), method='bounded')
#     a = res.x
#     b = (total_ab - a) / (k - 1)
#     return a, b

def snr_objective(a, s, k, total_ab):
    b = (total_ab - a) / (k - 1)
    if a <= b or b <= 0:  # 强制 a > b 且 b > 0
        return 1e9
    snr = ((a - b) ** 2) / (k * (a + (k - 1) * b))
    return abs(snr - s)

def find_a_given_snr(s, k, total_ab):
    # 设置 a 的范围为 (total_ab / k + ε, total_ab - ε)，保证 a > b 且 a < total_ab
    eps = 1e-4
    lower_bound = total_ab / k + eps
    upper_bound = total_ab - eps

    if lower_bound >= upper_bound:
        raise ValueError("No feasible solution: cannot satisfy a > b under given total_ab and k")

    res = minimize_scalar(
        snr_objective, args=(s, k, total_ab),
        bounds=(lower_bound, upper_bound), method='bounded'
    )
    a = res.x
    b = (total_ab - a) / (k - 1)
    return a, b