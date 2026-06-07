# Chapter 7 — 通勤时间分布可视化（理解分布型价值函数的前置直觉）
# 核心问题：我们习惯用"期望值"（平均值）做决策，但分布有时更合理
# 例子：开车均值 30 分钟但偶尔 90 分钟，火车均值 40 分钟但最坏也就 60 分钟
# 如果用均值 → 选开车；如果考虑方差/风险 → 选火车
# 这是 Ch7 后续"分布式 DQN (C51)"的动机——不学单点 Q 值，而学 Q 值的概率分布
import numpy as np
import matplotlib as mpl
mpl.use("Agg")                          # 非交互式后端，无需 GUI，直接保存图片
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 开车：均值 30 分钟，标准差 2，但有小概率（200/2200≈9%）花 90 分钟
    plt.clf()
    v1 = np.random.normal(30, 2.0, size=2000)   # 常态：30 分钟左右
    v2 = np.random.normal(90, 4.0, size=200)    # 偶发堵车：90 分钟
    v = np.concatenate((v1, v2))
    mean_time = v.mean()
    plt.hist(v, normed=True, bins=100)
    plt.title("Car commute time distribution\nmean=%.2f mins" % mean_time)
    plt.xlabel("Time, minutes")
    plt.ylabel("Probability")
    plt.savefig("commute-car.png")

    # 火车：均值 40 分钟，标准差 2，偶发 60 分钟（但概率极低，50/2050≈2.4%）
    plt.clf()
    v1 = np.random.normal(40, 2.0, size=2000)   # 常态：40 分钟
    v2 = np.random.normal(60, 1.0, size=50)     # 偶发延误：60 分钟
    v = np.concatenate((v1, v2))
    mean_time = v.mean()
    plt.hist(v, normed=True, bins=100)
    plt.title("Train commute time distribution\nmean=%.2f mins" % mean_time)
    plt.xlabel("Time, minutes")
    plt.ylabel("Probability")
    plt.savefig("commute-train.png")
