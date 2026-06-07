#!/usr/bin/env python3
# Chapter 7 — 普通回放缓冲区性能基准测试
# 对比 deque（双端队列）vs 循环数组 list 两种实现
# 测试 append / sample(4/8/16/32) 在不同容量下的吞吐量
import timeit
import numpy as np
import collections


SIZES = [10**n for n in (3, 4, 5)]         # 测试容量：1K / 10K / 100K
DATA_SHAPE = (84, 84, 4)                    # Atari 帧堆叠（uint8 节省内存）
REPEAT_NUMBER = 10                          # 每个测试重复次数


class ExperienceBufferDeque:
    """回放缓冲区 — deque 实现
    优点：代码简洁，deque 自动处理容量限制
    缺点：deque 按索引取值 O(n)，采样大量数据时慢
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)       # 满了自动 pop 最旧

    def sample(self, batch_size):
        # 随机索引 → 取值（deque[index] 每次 O(n)，总 O(n·batch_size)）
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[idx] for idx in indices]


class ExperienceBufferCircularList:
    """回放缓冲区 — 循环数组 list 实现
    优点：list 按索引取值 O(1)，采样更快
    缺点：需要手动管理循环指针和容量
    """
    def __init__(self, capacity):
        self.buffer = list()
        self.capacity = capacity
        self.pos = 0                           # 循环写入位置

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)     # 未满：直接追加
        else:
            self.buffer[self.pos] = experience # 已满：覆盖最旧位置
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[idx] for idx in indices]


def fill_buf(buf, size):
    """用零数组填满缓冲区（模拟存储 Atari 帧）"""
    for _ in range(size):
        buf.append(np.zeros(DATA_SHAPE, dtype=np.uint8))


def bench_buffer(buf_class):
    """对缓冲区类跑标准性能测试"""
    print("Benchmarking %s" % buf_class.__name__)

    for size in SIZES:
        print("  Test size %d" % size)
        ns = globals()
        ns.update(locals())
        # 初始填充：从空到满
        t = timeit.timeit('fill_buf(buf, size)',
                         setup='buf = buf_class(size)',
                         number=REPEAT_NUMBER, globals=ns)
        print("  * Initial fill:\t%.2f items/s" % (size*REPEAT_NUMBER / t))
        # Append 速度：满容量后继续写入
        buf = buf_class(size)
        fill_buf(buf, size)
        ns.update(locals())
        t = timeit.timeit('fill_buf(buf, size)',
                         number=REPEAT_NUMBER, globals=ns)
        print("  * Append:\t\t%.2f items/s" % (size*REPEAT_NUMBER / t))
        # 不同 batch_size 下的采样速度（RL 常用 32/64）
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
    bench_buffer(ExperienceBufferCircularList)
    bench_buffer(ExperienceBufferDeque)
    pass
