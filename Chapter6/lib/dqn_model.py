# DQN 网络结构（Atari 版本）— 来自 DeepMind 2015 Nature 论文
# 与 Chapter 4 策略网络的区别：输出层不经过 softmax，直接输出原始 Q 值
import torch
import torch.nn as nn

import numpy as np


class DQN(nn.Module):
    """用于 Atari 游戏的卷积 Q 网络
    输入：(batch, 4, 84, 84) — 4 帧灰度图堆叠
    输出：(batch, n_actions) — 每个动作的 Q 值

    结构：3 层卷积 → flatten → 2 层全连接
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # 卷积层：从像素中提取空间特征
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # (4,84,84) → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # (32,20,20) → (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # (64,9,9) → (64,7,7)
            nn.ReLU()
        )

        # 计算卷积输出展平后的尺寸，用于全连接层输入
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),  # 展平后的特征 → 512
            nn.ReLU(),
            nn.Linear(512, n_actions)        # 512 → 动作数（Pong: 6 个动作）
        )

    def _get_conv_out(self, shape):
        """用全零张量跑一遍卷积，自动计算输出维度
        避免手动计算每层卷积后的尺寸
        """
        o = self.conv(torch.zeros(1, *shape))  # (1, C, H, W)
        return int(np.prod(o.size()))            # 展平后的总元素数

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)  # 卷积 → 展平为 (batch, features)
        return self.fc(conv_out)                        # 全连接 → (batch, n_actions)
