#!/usr/bin/env python3
# Chapter 6 — DQN 玩 Atari Pong
# 本章核心：将 DQN 从 CartPole（4 维向量）扩展到 Atari（像素输入）
# 关键变化：
#   1. Q 网络：全连接 → 卷积神经网络（CNN）
#   2. 环境预处理：原始 Atari 帧 → 灰度 84×84 → 4 帧堆叠
#   3. 经验回放 + 目标网络 + ε 衰减（与 CartPole 版本相同）
from lib import wrappers   # Atari 环境预处理
from lib import dqn_model  # CNN Q 网络

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"  # Pong：双人乒乓球
MEAN_REWARD_BOUND = 19.5                  # 最近 100 回合平均奖励 ≥ 19.5 即停

# === 超参数 ===
GAMMA = 0.99                 # 折扣因子
BATCH_SIZE = 32              # 每次训练采样的经验数
REPLAY_SIZE = 10000           # 回放缓冲区容量
LEARNING_RATE = 1e-4          # Adam 学习率
SYNC_TARGET_FRAMES = 1000     # 每 1000 帧同步一次目标网络
REPLAY_START_SIZE = 10000     # 缓冲区至少收集这么多条经验才开始训练

EPSILON_DECAY_LAST_FRAME = 10**5  # ε 从 1.0 线性衰减到 0.02 的总帧数
EPSILON_START = 1.0               # 初始 ε：100% 随机探索
EPSILON_FINAL = 0.02              # 最终 ε：2% 随机探索


# 经验元组：比 CartPole 版本多了 done（是否为终止状态）
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    """经验回放缓冲区（与 Ch6 笔记本逻辑一致，实现略有不同）
    固定容量 deque，满了自动覆盖最旧的
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """随机采样，返回 numpy 数组"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # 按索引取出 → zip 解包 → 分别转为 numpy 数组
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices])
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.uint8),
                np.array(next_states))


class Agent:
    """DQN Agent（Atari 版）
    负责与环境交互，按 ε-greedy 选动作，将经验存入缓冲区
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        """每回合开始时重置状态和累计奖励"""
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """执行一步环境交互（ε-greedy）
        参数：
          net: Q 网络（用于贪心动作选择）
          epsilon: 探索概率
        返回：
          如果回合结束 → 返回累计奖励；否则 → 返回 None
        """
        done_reward = None

        # ε-greedy 动作选择
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()  # 随机探索
        else:
            state_a = np.array([self.state], copy=False)        # (84,84,4) → (1,84,84,4)
            state_v = torch.tensor(state_a).to(device)          # 转张量 + 送 GPU
            q_vals_v = net(state_v)                              # 前向传播得 Q 值
            _, act_v = torch.max(q_vals_v, dim=1)                # 取最大 Q 值的索引
            action = int(act_v.item())

        # 执行动作
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # 存入经验（注意：存的是原始 state，不是预处理后的）
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()  # 回合结束，重置环境
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    """计算 DQN 损失（TD 误差的 MSE）
    batch: (states, actions, rewards, dones, next_states)
    net: 策略网络（需要训练）
    tgt_net: 目标网络（不参与梯度，用于计算稳定的 TD 目标）
    """
    states, actions, rewards, dones, next_states = batch

    # 转为 GPU 张量
    states_v = torch.tensor(states).to(device)            # (B, 4, 84, 84)
    next_states_v = torch.tensor(next_states).to(device)  # (B, 4, 84, 84)
    actions_v = torch.tensor(actions).to(device)           # (B,)
    rewards_v = torch.tensor(rewards).to(device)           # (B,)
    done_mask = torch.ByteTensor(dones).to(device)         # (B,) 1=终止

    # 当前 Q(s, a)：策略网络输出中取出实际执行动作的值
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)  # (B,)

    # TD 目标：r + γ * max_a' Q_target(s', a') * (1 - done)
    next_state_values = tgt_net(next_states_v).max(1)[0]   # 目标网络的最大 Q 值
    next_state_values[done_mask] = 0.0                      # 终止状态未来价值为 0
    next_state_values = next_state_values.detach()          # 阻断梯度

    expected_state_action_values = next_state_values * GAMMA + rewards_v  # Bellman 方程
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # ① 创建预处理后的环境 + 双网络 + 回放缓冲区
    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []   # 每回合的累计奖励
    frame_idx = 0         # 总帧数计数器
    ts_frame = 0          # 上一次打印时的帧数（用于算 FPS）
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        # ② ε 线性衰减
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        # ③ 与环境交互一步
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            # 回合结束，记录并评估
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])  # 最近 100 回合平均
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            # 保存最佳模型
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (
                        best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        # ④ 缓冲区不够大先跳过训练
        if len(buffer) < REPLAY_START_SIZE:
            continue

        # ⑤ 定期同步目标网络
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # ⑥ 采样 + 训练一步
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
