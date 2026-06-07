#!/usr/bin/env python3
# Chapter 7 — 优先级回放缓冲区性能基准测试
# 对比两种实现：deque（双端队列）vs list（循环数组）
# 在不同容量（1000/1万/10万）下测试 append / sample(4/8/16/32) 的速度
import timeit
import numpy as np
import collections

SIZES = [10**n for n in (3, 4, 5)]         # 测试容量：1K / 10K / 100K
DATA_SHAPE = (84, 84, 4)                    # Atari 帧堆叠的形状
REPEAT_NUMBER = 10                          # 每个测试重复次数


class PrioReplayBufferDeque:
    """优先级回放 — deque 实现
    双端队列：append 快（O(1)），但采样需要先取索引再按索引取值
    """
    def __init__(self, buf_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha          # 优先级指数：0=均匀采样，1=按优先级采样
        self.buffer = collections.deque(maxlen=buf_size)
        self.priorities = collections.deque(maxlen=buf_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, sample):
        # 新经验的优先级 = 当前最大优先级（刚开始为 1.0）
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(sample)
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        # 按优先级概率采样：p(i) = prio(i)^α / Σ prio(j)^α
        probs = np.array(self.priorities, dtype=np.float32) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        # 重要性采样权重（IS weights）：补偿非均匀采样引入的偏差
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()              # 归一化到 [0,1]
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioReplayBufferList:
    """优先级回放 — list + 循环数组实现
    用固定大小 list + 循环指针替代 deque，避免 deque 的索引开销
    """
    def __init__(self, buf_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0                           # 循环指针
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, sample):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)          # 未满：追加
        else:
            self.buffer[self.pos] = sample      # 已满：覆盖最旧的
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # 只考虑已填充部分（未满时只有前 pos 个位置有效）
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


def fill_buf(buf, size):
    """用零数组填满缓冲区"""
    for _ in range(size):
        buf.append(np.zeros(DATA_SHAPE, dtype=np.uint8))


def bench_buffer(buf_class):
    """对缓冲区类跑标准性能测试"""
    print("Benchmarking %s" % buf_class.__name__)

    for size in SIZES:
        print("  Test size %d" % size)
        ns = globals()
        ns.update(locals())
        # 初始填充速度
        t = timeit.timeit('fill_buf(buf, size)',
                         setup='buf = buf_class(size)',
                         number=REPEAT_NUMBER, globals=ns)
        print("  * Initial fill:\t%.2f items/s" % (size*REPEAT_NUMBER / t))
        # Append 速度（满容量后继续 append 即覆盖）
        buf = buf_class(size)
        fill_buf(buf, size)
        ns.update(locals())
        t = timeit.timeit('fill_buf(buf, size)',
                         number=REPEAT_NUMBER, globals=ns)
        print("  * Append:\t\t%.2f items/s" % (size*REPEAT_NUMBER / t))
        # 不同 batch_size 的采样速度
        t = timeit.timeit('buf.sample(4)',
                         number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 4:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))
        t = timeit.timeit('buf.sample(8)',
                         number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 8:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))
        t = timeit.timeit('buf.sample(16)',
                         number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 16:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))
        t = timeit.timeit('buf.sample(32)',
                         number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 32:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))


if __name__ == "__main__":
    bench_buffer(PrioReplayBufferList)
    bench_buffer(PrioReplayBufferDeque)
    pass
