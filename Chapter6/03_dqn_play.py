#!/usr/bin/env python3
# Chapter 6 — 使用训练好的 DQN 模型玩游戏（纯推理，不训练）
# 加载保存的模型权重，用贪心策略（ε=0）玩一局，可选录制视频
import gym
import time
import argparse
import numpy as np

import torch

from Chapter6.lib import wrappers
from Chapter6.lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25  # 渲染帧率（Pong 正常速度）


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory to store video recording")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization of the game play")
    args = parser.parse_args()

    # ① 创建环境（与训练时相同的预处理流水线）
    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)  # 录制视频

    # ② 加载训练好的模型
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    # 加载到 CPU（map_location 确保跨设备兼容）

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()  # 统计每个动作被选中的次数

    while True:
        start_ts = time.time()
        if args.visualize:
            env.render()  # 显示游戏画面

        # ③ 选动作：纯贪心（ε=0），始终选 Q 值最大的动作
        state_v = torch.tensor(np.array([state], copy=False))  # 添加 batch 维度
        q_vals = net(state_v).data.numpy()[0]                  # Q 值数组
        action = np.argmax(q_vals)                             # 取最大 Q 值的动作
        c[action] += 1

        # ④ 执行动作
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

        # ⑤ 控制帧率（让画面以正常速度播放）
        if args.visualize:
            delta = 1/FPS - (time.time() - start_ts)  # 本帧还差多少时间
            if delta > 0:
                time.sleep(delta)

    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)  # 查看 Agent 的动作偏好分布
    if args.record:
        env.env.close()
