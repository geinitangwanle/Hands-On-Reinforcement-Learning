# Chapter 7 — 分布式价值投影测试（C51 算法的核心操作可视化）
# C51 论文："A Distributional Perspective on Reinforcement Learning"
# 核心思想：不用单点值表示 Q(s,a)，而用一个 51 原子（atom）的离散分布
# 这个脚本测试 distr_projection 函数：如何将下一状态的分布"投影"回当前状态
# 即：下一状态的 Q 值分布 → 经 Bellman 方程转换 → 当前状态的 Q 值分布
import sys
import numpy as np
sys.path.append("./")

from lib import common                            # 导入 distr_projection 函数

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


Vmax = 10                                         # 价值上界（原子覆盖范围 [-10, 10]）
Vmin = -10                                        # 价值下界
N_ATOMS = 51                                      # 原子数（51 个离散 bin）
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)            # 每个原子的宽度 = 0.4


def save_distr(src, proj, name):
    """保存分布对比图：原始的下一状态分布 vs 投影后的当前状态分布"""
    plt.clf()
    p = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)    # x 轴：价值 [-10, -9.6, ..., 9.6, 10]
    plt.subplot(2, 1, 1)
    plt.bar(p, src, width=0.5)
    plt.title("Source")
    plt.subplot(2, 1, 2)
    plt.bar(p, proj, width=0.5)
    plt.title("Projected")
    plt.savefig(name + ".png")


if __name__ == "__main__":
    np.random.seed(123)
    atoms = np.arange(Vmin, Vmax+DELTA_Z, DELTA_Z)

    # 测试 1：单峰分布 + r=2，非终止
    # 某个原子概率为 1.0，经 Bellman 投影后分布如何变化？
    src_hist = np.zeros(shape=(1, N_ATOMS), dtype=np.float32)
    src_hist[0, N_ATOMS//2+1] = 1.0               # 仅中间原子有概率
    proj_hist = common.distr_projection(
        src_hist, np.array([2], dtype=np.float32), np.array([False]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist[0], proj_hist[0], "peak-r=2")

    # 测试 2：正态分布 + r=2，非终止
    # 更现实的分布形状，验证投影保形性
    data = np.random.normal(size=1000, scale=3)
    hist = np.histogram(data, normed=True,
                        bins=np.arange(Vmin - DELTA_Z/2, Vmax + DELTA_Z*3/2, DELTA_Z))
    src_hist = hist[0]
    proj_hist = common.distr_projection(
        np.array([src_hist]), np.array([2], dtype=np.float32), np.array([False]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_hist[0], "normal-r=2")

    # 测试 3：正态分布 + r=2，终止状态（done=True）
    # 终止时分布应该被重置——只反映即时奖励，没有未来价值
    proj_hist = common.distr_projection(
        np.array([src_hist]), np.array([2], dtype=np.float32), np.array([True]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_hist[0], "normal-done-r=2")

    # 测试 4：奖励超出范围（r=10 > Vmax）——验证裁剪
    proj_dist = common.distr_projection(
        np.array([src_hist]), np.array([10], dtype=np.float32), np.array([False]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(hist[0], proj_dist[0], "normal-r=10")

    # 测试 5：batch 中混合 done/not-done —— 验证 done 掩码的正确性
    proj_hist = common.distr_projection(
        np.array([src_hist, src_hist]),
        np.array([2, 2], dtype=np.float32),
        np.array([False, True]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_not_clip-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_not_clip-02-complete")

    # 测试 6：batch 混合 + 右裁剪（r=10 超出 Vmax）
    proj_hist = common.distr_projection(
        np.array([src_hist, src_hist]),
        np.array([10, 10], dtype=np.float32),
        np.array([False, True]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_clip-right-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_clip-right-02-complete")

    # 测试 7：batch 混合 + 左裁剪（r=-10 超出 Vmin）
    proj_hist = common.distr_projection(
        np.array([src_hist, src_hist]),
        np.array([-10, -10], dtype=np.float32),
        np.array([False, True]),
        Vmin, Vmax, N_ATOMS, gamma=0.9)
    save_distr(src_hist, proj_hist[0], "both_clip-left-01-incomplete")
    save_distr(src_hist, proj_hist[1], "both_clip-left-02-complete")

    pass
