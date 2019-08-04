import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np
import random

MAX_LEN = 2000
BATCH_SIZE = 64
GAMMA = 0.95
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.1

class Agent(object):
    def __init__(self, input_space, output_space, lr=0.001, exploration=0.9):
        self._model = keras.Sequential()
        self._model.add(keras.layers.Dense(input_shape=(input_space,), units=24, activation=tf.nn.relu))
        self._model.add(keras.layers.Dense(units=24, activation=tf.nn.relu))
        # 注意这里输出层的激活函数是线性的！！！
        self._model.add(keras.layers.Dense(units=output_space, activation='linear'))
        self._model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr))

        self._replayBuffer = deque(maxlen=MAX_LEN)  # replay buffer，最大200的容量
        self._exploration = exploration

    @property
    def exploration(self):
        return self._exploration

    def add_data(self, state, action, reward, state_next, done):
        self._replayBuffer.append((state, action, reward, state_next, done))

    def act(self, state):
        if np.random.uniform() <= self._exploration:  # 随机走出一步
            return np.random.randint(0, 2)
        action = self._model.predict(state)  # 使用神经网络评估的选择
        return np.argmax(action[0])

    def train_from_buffer(self):
        if len(self._replayBuffer) < BATCH_SIZE:
            return
        batch = random.sample(self._replayBuffer, BATCH_SIZE)  # 随机选取一个批次的数据
        for state, action, reward, state_next, done in batch:
            if done:  # 对应论文中的分数更新
                q_update = reward
            else:
                q_update = reward + GAMMA * np.amax(self._model.predict(state_next)[0])
            q_values = self._model.predict(state)  # 先赋值，为了减去不相关的行动得分
            q_values[0][action] = q_update  # 把采取了的行动的分数更新，那么只有这项在MSE中有效果
            self._model.fit(state, q_values, verbose=0)  # SGD训练模型
            self._exploration *= EXPLORATION_DECAY
            self._exploration = max(EXPLORATION_MIN, self._exploration)
