#!/usr/bin/env python3
# Chapter 7 — DQN 改进：ptan 库版本（Basic DQN）
# 与 Ch6 的核心区别：用 ptan 库封装 EpsilonGreedy / ExperienceSource / TargetNet / ReplayBuffer
# 原来手写的 Agent / ReplayBuffer / 训练循环 → 现在用 ptan 提供的标准组件
# 好处：代码量大幅减少，组件可复用，方便后续扩展（Double DQN、Dueling 等）
import gym
import ptan                                        # PyTorch Agent Net — RL 训练工具库
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common                   # 本地模块：网络定义 + 工具函数


if __name__ == "__main__":
    # ① 加载超参数 — 统一从 common.HYPERPARAMS 读取，支持多个游戏
    params = common.HYPERPARAMS['pong']
#    params['epsilon_frames'] = 200000               # 可调：ε 衰减更快，提早进入利用阶段
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # ② 创建环境 + ptan 标准预处理包装器
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)        # 一键套用 Ch6 那 6 个 wrapper（跳帧/灰度/CHW/堆叠/归一化）

    writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    # ③ ptan 封装的关键组件
    tgt_net = ptan.agent.TargetNet(net)              # 目标网络 — 替代手写 load_state_dict 同步
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params['epsilon_start'])             # ε-greedy 动作选择器 — 替代手写 select_action
    epsilon_tracker = common.EpsilonTracker(selector, params)  # ε 线性衰减跟踪器
    agent = ptan.agent.DQNAgent(net, selector, device=device)  # DQN Agent — 封装了网络 + 动作选择

    # ④ 经验源 + 回放缓冲区（ptan 封装）
    # ExperienceSourceFirstLast：自动执行 step → 收集 (s,a,r,s',done) 经验
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params['replay_size'])

    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    # ⑤ 奖励追踪器（上下文管理器，自动记录 + 打印 + 判断停止条件）
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            # ⑥ 从环境采集 1 步经验 → 存入缓冲区
            buffer.populate(1)
            # ⑦ 更新 ε 值（线性衰减）
            epsilon_tracker.frame(frame_idx)

            # ⑧ 检查是否有回合结束的奖励
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                # 打印日志 + 判断是否达到停止条件
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            # ⑨ 缓冲区不够大，跳过训练（先收集 REPLAY_INITIAL 条经验）
            if len(buffer) < params['replay_initial']:
                continue

            # ⑩ 采样 → 算损失 → 梯度更新
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(
                batch, net, tgt_net.target_model,
                gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            # ⑪ 定期同步目标网络（每 TARGET_NET_SYNC 帧）
            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
