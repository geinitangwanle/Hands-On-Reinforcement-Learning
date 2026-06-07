# Atari 环境包装器 — 将原始 Atari 帧转换为 DQN 可用的格式
# 每条包装器负责一个独立的预处理步骤，可自由组合
# 参考：DeepMind 2015 Nature DQN 论文的预处理流程
import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    """处理需要按 FIRE 才能开始的游戏（如 Pong、Breakout）
    在 reset 时自动按下 FIRE（动作 1），让游戏进入进行中状态
    """
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """重置环境 → 按 FIRE 开始游戏 → 按 FIRE 发球
        如果中途结束则再次重置（处理 FIRE 后立即死亡的情况）
        """
        self.env.reset()
        obs, _, done, _ = self.env.step(1)  # 按 FIRE 开始
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)  # 按 FIRE 发球（Pong 特有）
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """跳帧 + 像素最大值合并
    每 skip 帧返回一帧（默认 skip=4），只做 skip 次动作决策
    取最近两帧的逐像素最大值，消除闪烁效果
    """
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)  # 保留最近 2 帧用于 max
        self._skip = skip

    def step(self, action):
        """同一动作执行 skip 次，累积奖励，返回 max 后的帧"""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        # 取最近两帧的逐像素最大值，消除奇数/偶数帧闪烁
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """清空缓冲区，返回初始帧"""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """帧预处理：彩色 → 灰度 → resize 到 84×84
    DeepMind 论文的标准预处理：
      1. 提取亮度通道（Y = 0.299R + 0.587G + 0.114B）
      2. 缩放到 84×110
      3. 裁剪顶部 18 像素（去除比分牌），得到 84×84
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        """静态方法：可独立调用，将原始帧转为 84×84 灰度图"""
        # 判断原始分辨率并重塑
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        # RGB → 灰度（ITU-R BT.601 亮度公式）
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # 缩放并裁剪
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]  # 去掉顶部比分牌（18 像素）和底部（8 像素）
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """维度重排：HWC → CHW
    numpy 图像默认 (H, W, C)，PyTorch 卷积期望 (C, H, W)
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        # 新的 observation_space：(C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)  # (84,84,1) → (1,84,84)


class ScaledFloatFrame(gym.ObservationWrapper):
    """像素值归一化：uint8 [0,255] → float32 [0.0, 1.0]
    有助于神经网络训练的数值稳定性
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """帧堆叠：将最近 n_steps 帧沿通道维度堆叠
    DQN 需要多帧输入才能感知运动（单帧无法判断速度方向）
    Atari 论文中 n_steps=4，即每步输入最近 4 帧灰度图
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        # 新的 observation_space：沿通道维重复 n_steps 倍
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        """重置缓冲区：所有帧初始化为 0"""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """滑动窗口：丢弃最旧帧，添加最新帧"""
        self.buffer[:-1] = self.buffer[1:]       # 左移一位，丢弃最旧的
        self.buffer[-1] = observation            # 最新帧放末尾
        return self.buffer


def make_env(env_name):
    """创建完整的预处理环境流水线
    按顺序套用所有包装器：
      原始 Atari → MaxAndSkip(跳帧) → FireReset(开始游戏)
      → Frame84(灰度+缩放) → ImageToPyTorch(CHW)
      → BufferWrapper(堆叠4帧) → ScaledFloat(归一化)
    最终输出：(4, 84, 84) 的 float32 张量
    """
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)        # 每 4 帧决策一次，帧间 max 合并
    env = FireResetEnv(env)         # 自动按 FIRE 开始游戏
    env = ProcessFrame84(env)       # 灰度化 + 84×84 缩放
    env = ImageToPyTorch(env)       # CHW 格式转换
    env = BufferWrapper(env, 4)     # 堆叠最近 4 帧
    return ScaledFloatFrame(env)    # [0, 1] 归一化
