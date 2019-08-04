import gym
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt


def train():
    env = gym.make("CartPole-v1")
    input_space = env.observation_space.shape[0]
    output_space = env.action_space.n
    print(input_space, output_space)
    agent = Agent(input_space, output_space)
    run = 0
    x = []
    y = []
    while run < 100:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, -1])
        step = 0
        while True:
            step += 1  # 步数越多，相当于站立的时间越长，比较容易理解。
            env.render()
            action = agent.act(state)
            state_next, reward, done, _ = env.step(action)
            reward = reward if not done else -reward  # 棍子倒了，分数肯定是负数了
            state_next = np.reshape(state_next, [1, -1])
            agent.add_data(state, action, reward, state_next, done)
            state = state_next
            if done:
                print("Run: " + str(run) + ", exploration: " +
                      str(agent.exploration) + ", score:" + str(step))
                x.append(run)
                y.append(step)
                break
            agent.train_from_buffer()  # 每次都要执行训练
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    train()