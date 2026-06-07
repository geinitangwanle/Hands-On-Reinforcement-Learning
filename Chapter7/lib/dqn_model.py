# Chapter 7 — DQN 网络定义（Atari 版 + NoisyNet 扩展）
# 包含三个类：NoisyLinear（独立噪声）/ NoisyFactorizedLinear（因子化噪声）/ DQN（标准 CNN）
# 前两者用于 Ch7 后续的 Noisy DQN 变体，DQN 是当前 Basic DQN 使用的标准网络
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NoisyLinear(nn.Linear):
    """NoisyNet 噪声线性层（独立噪声版本）
    思路：将网络参数变成"可学习的均值 + 可学习的方差 × 随机噪声"
    每次 forward 重新采样噪声，自动实现探索，不需要 ε-greedy
    公式：y = (W_mean + W_sigma ⊙ ε_W) · x + b_mean + b_sigma ⊙ ε_b
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # sigma_weight：可学习的噪声标准差（权重部分）
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init))
        # epsilon_weight：每次 forward 采样的随机噪声（buffer，不参与梯度）
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        # 每次 forward 重新采样高斯噪声
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        # W_effective = W_mean + W_sigma ⊙ ε_W
        return F.linear(input,
                        self.weight + self.sigma_weight * self.epsilon_weight.data,
                        bias)


class NoisyFactorizedLinear(nn.Linear):
    """NoisyNet 因子化噪声版本（更高效，论文推荐）
    不独立采样每个权重噪声，而是用两个低维噪声向量外积：ε_in ⊗ ε_out
    参数量从 O(in×out) 降到 O(in+out)
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init))
        # 因子化噪声：两个小向量替代完整噪声矩阵
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        # 因子化变换：f(x) = sign(x) * √|x| （与论文一致）
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)       # (out, in) = (out,1) × (1,in)
        return F.linear(input,
                        self.weight + self.sigma_weight * noise_v,
                        bias)


class DQN(nn.Module):
    """标准 CNN DQN（与 Ch6 结构一致，多了 div 256 归一化）
    输入：(batch, 4, 84, 84) uint8
    输出：(batch, n_actions) Q 值
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # 3 层卷积：从 4 帧灰度图中提取空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # (C,84,84)→(32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # →(64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # →(64,7,7)
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)              # Pong: 6 动作; Breakout: 4 动作
        )

    def _get_conv_out(self, shape):
        """用零张量跑一遍卷积，自动计算展平后的特征数"""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # 关键变化：先归一化（uint8 [0,255] → float [0,1]），比 Ch6 的 wrapper 内归一化更灵活
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)  # 卷积 → 展平
        return self.fc(conv_out)
