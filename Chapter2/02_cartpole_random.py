import gym 

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    total_reward = 0.0
    total_steps = 0
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    while True:
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, reward, done, _ = step_out
        total_reward += reward
        total_steps += 1
        if done:
            break
    
    print("Episode done in %d steps, total reward %.2f"%(total_steps, total_reward))
    
