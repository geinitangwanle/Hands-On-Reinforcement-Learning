# Chapter 7 公共工具模块
# 包含：超参数字典 / 经验解包 / DQN 损失 / 奖励追踪 / ε 衰减 / 分布式投影
import sys
import time
import numpy as np
import torch
import torch.nn as nn


# 各游戏超参数字典 — 从"散落全局变量"变成"集中管理的数据结构"
# 后续代码只需 import 这个字典，不用在每个文件里重复定义
HYPERPARAMS = {
    'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,                # 最近 100 局平均奖励 ≥ 18 即停
        'run_name':         'pong',
        'replay_size':      100000,              # 缓冲区容量（比 Ch6 的 1 万大 10 倍）
        'replay_initial':   10000,               # 先收集 1 万条经验再开始训练
        'target_net_sync':  1000,                # 每 1000 帧同步目标网络
        'epsilon_frames':   10**5,               # ε 从 1.0 → 0.02 衰减的帧数
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    },
    'breakout-small': {                           # Breakout 小版本：缓冲区 30 万
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,                 # 最终 ε 更高（0.1 vs 0.02），保持更多探索
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    },
    'breakout': {                                 # Breakout 完整版：缓冲区 100 万
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,              # 百万级缓冲区
        'replay_initial':   50000,               # 先收集 5 万条经验
        'target_net_sync':  10000,               # 同步间隔更长（1 万帧）
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    },
    'invaders': {                                 # 太空侵略者
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
}


def unpack_batch(batch):
    """将 ptan Experience 对象列表 → 分别的 numpy 数组
    ptan 的 Experience 包含 .state / .action / .reward / .last_state
    last_state 为 None 表示这是终止状态（没有下一状态）
    此时用当前 state 填充（反正会被 done 掩码屏蔽）
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)           # 无下一状态 → done=True
        if exp.last_state is None:
            last_states.append(state)                  # 占位填充（后续被 done 掩码清 0）
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return (np.array(states, copy=False),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(last_states, copy=False))


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    """标准 DQN 损失：MSE(Q(s,a), r + γ·max Q_target(s',a') · (1-done))
    与 Ch6 手写版逻辑完全一致，但通过 unpack_batch 适配 ptan 的数据格式
    """
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)          # (B, 4, 84, 84)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)         # (B,)
    rewards_v = torch.tensor(rewards).to(device)         # (B,)
    done_mask = torch.ByteTensor(dones).to(device)       # (B,) 1=终止

    # 当前 Q(s,a)：策略网络输出中取出实际执行动作的 Q 值
    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)          # (B,)
    # TD 目标：r + γ·max Q_target(s',a')，终止时未来清零
    next_state_values = tgt_net(next_states_v).max(1)[0] # max Q_target(s', a')
    next_state_values[done_mask] = 0.0                    # done → 未来无价值
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class RewardTracker:
    """奖励追踪器（上下文管理器）
    自动记录奖励、打印日志、写 TensorBoard、判断是否到停止条件
    """
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []                          # 所有回合的累计奖励
        return self

    def __exit__(self, *args):
        self.writer.close()                              # 退出时关闭 TensorBoard writer

    def reward(self, reward, frame, epsilon=None):
        """每回合结束时调用
        返回 True → 达到停止条件，训练结束
        返回 False → 继续训练
        """
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)  # 计算 FPS
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])   # 最近 100 局平均
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class EpsilonTracker:
    """ε 线性衰减控制器
    不负责动作选择，只负责根据帧数更新 EpsilonGreedyActionSelector 的 ε 值
    """
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        """每帧调用，线性衰减 ε"""
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final,
                self.epsilon_start - frame / self.epsilon_frames)


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """C51 分布式强化学习的核心操作：分布投影
    将下一状态的 Q 值概率分布，经 Bellman 方程投影为当前状态的 Q 值分布
    输入：
      next_distr: (batch, n_atoms) 下一状态的概率分布
      rewards:    (batch,) 即时奖励
      dones:      (batch,) 是否终止
      Vmin/Vmax:  价值范围，等距原子 z_i = Vmin + i · Δz
      gamma:      折扣因子
    输出：
      proj_distr: (batch, n_atoms) 投影后的当前状态分布
    算法原理：
      Bellman 方程作用于每个原子 → T_z_j = r + γ·z_j
      然后分配到最近的两个 bin → 类似双线性插值
      终止状态：分布退化为单一 Delta 函数（只反映即时奖励）
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    # 非终止状态：Bellman 投影
    for atom in range(n_atoms):
        # 该原子的投影位置：T_z_j = r + γ·z_j，限制在 [Vmin, Vmax] 内
        tz_j = np.minimum(Vmax, np.maximum(
            Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z             # 浮点位置（如 25.3）
        l = np.floor(b_j).astype(np.int64)         # 左 bin 索引
        u = np.ceil(b_j).astype(np.int64)          # 右 bin 索引
        # 恰好落在 bin 中心 → 全部分配给该 bin
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        # 落在两个 bin 之间 → 按距离分配
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    # 终止状态：分布退化为即时奖励处的 Delta 函数
    if dones.any():
        proj_distr[dones] = 0.0                    # 清空原分布
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0  # 恰好落在 bin 中心 → 概率 1
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr
