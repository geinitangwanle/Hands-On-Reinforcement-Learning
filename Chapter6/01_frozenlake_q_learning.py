#!/usr/bin/env python3
# Chapter 6 — 表格型 Q-Learning（FrozenLake 冰湖问题）
# 核心思想：维护一张 Q(s,a) 表格，用 Bellman 方程迭代更新，无需神经网络
# 与 DQN 的关系：DQN = Q-Learning + 神经网络函数近似 + 经验回放 + 目标网络
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"  # 4×4 冰湖：走到终点 +1，掉冰窟窿 0
GAMMA = 0.9                 # 折扣因子：未来奖励的衰减率
ALPHA = 0.2                 # 学习率：新信息覆盖旧信息的比例（0=不学，1=只信最新）
TEST_EPISODES = 20          # 每轮评估跑多少回合取平均


class Agent:
    """表格型 Q-Learning Agent
    用 Python 字典存储 Q 值表：key=(state, action), value=Q值
    未曾访问过的 (s,a) 默认 Q=0（通过 defaultdict(float) 实现）
    """
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)  # Q 值表，默认 0.0

    def sample_env(self):
        """随机采样一步（纯探索，不依赖 Q 表）
        返回一条经验：(状态, 动作, 奖励, 下一状态)
        """
        action = self.env.action_space.sample()  # 随机选动作（上下左右）
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state  # 结束后重置
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):
        """给定状态，找 Q 值最大的动作及其 Q 值
        遍历所有动作，返回 (max_Q_value, best_action)
        """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        """Q-Learning 核心更新公式（指数移动平均）：
        Q(s,a) ← (1-α)·Q(s,a) + α·[r + γ·max_a' Q(s',a')]
        其中：
          r + γ·max_a' Q(s',a') 是 TD 目标（Bellman 最优方程）
          α 控制新旧信息的混合比例
        """
        best_v, _ = self.best_value_and_action(next_s)       # max_a' Q(s', a')
        new_val = r + GAMMA * best_v                          # TD 目标
        old_val = self.values[(s, a)]                         # 当前 Q 值
        self.values[(s, a)] = old_val * (1-ALPHA) + new_val * ALPHA  # 软更新

    def play_episode(self, env):
        """评估用：用当前 Q 表纯贪心地跑一个回合（无探索）"""
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)  # 只选最优动作
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        # ① 随机采样一步环境交互
        s, a, r, next_s = agent.sample_env()
        # ② Q 值表格更新
        agent.value_update(s, a, r, next_s)

        # ③ 每轮评估：用当前 Q 表跑 20 回合，取平均成功率
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:  # 80% 成功率即算解决
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
