import numpy as np
from collections import deque
import random

class DQNAgent:

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.lr = 0.01
        self.memory = deque(maxlen=1000)
        self.q = np.zeros(shape=(state_dim,action_dim))
        self.exploration = 0
        self.exploitation = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            self.exploration += 1
            return np.random.choice(self.action_dim)
        self.exploitation += 1
        return np.argmax(self.q[state])

    def infer(self,state):
        return np.argmax(self.q[state])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        scalar_loss = 0
        for state, action, reward, next_state, done in minibatch:
            if not done:
                self.q[state,action] += self.lr*(reward + self.gamma * np.max(self.q[next_state])) - self.q[state, action]
#                self.q[state, action] = reward + self.gamma * np.max(self.q[next_state])
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        return scalar_loss

    def save_llm(self):
        np.save(file='llm_taxi.npy', arr=self.q)

    def load_llm(self):
        self.q = np.load('llm_taxi.npy')

    def save(self):
        np.save(file='rl_taxi.npy', arr=self.q)

    def load(self):
        self.q = np.load('rl_taxi.npy')