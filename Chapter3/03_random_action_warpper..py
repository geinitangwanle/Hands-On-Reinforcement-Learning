import gym
from typing import TypeVar
import random

Action = TypeVar('Action')

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon

    def action(self, action:Action) -> Action:
        if random.random()<self.epsilon:
            print("random!")
            return self.env.action_space.sample()
        return action
    
if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v0"))
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    total_reward = 0.0

    while True:
        step_out = env.step(0)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, reward, done, _ = step_out

        total_reward += reward
        if done:
            break

    print("Reward got :%.2f"% total_reward)
