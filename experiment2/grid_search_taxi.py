import math

import gymnasium as gym

import pandas as pd

from problem_solving.dqn_agent import DQNAgent


def train():
    env = gym.make('Taxi-v3', max_episode_steps=500, is_rainy=True, fickle_passenger=True)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    max_terminated = -math.inf
    optimal_agent = None
    import numpy as np
    for lr in np.arange(0.01,0.06,0.01):
        for g in np.arange(0.9,1.0,0.01):
            agent = DQNAgent(state_size,action_size, gamma=g,lr=lr)
            episodes = 100
            batch_size = 32
            count = 0
            for episode in range(episodes):
                state = env.reset()[0]
                truncated = False
                terminated = False
                steps = 0
                agent.exploitation = 0
                agent.exploration = 0
                while not truncated and not terminated:
                    action = agent.act(state)
                    observation, reward, terminated, truncated, info = env.step(action)
                    agent.remember(state, action, reward, observation, terminated)
                    state = observation
                    agent.replay(batch_size)
                    steps += 1
                if (terminated):
                    count += 1
            if (count > max_terminated):
                print('Updating optimal agent ',(lr,g,count,max_terminated))
                max_terminated = count
                optimal_agent = agent
    optimal_agent.save_grid()

def infer():
    env = gym.make('Taxi-v3', max_episode_steps=200, is_rainy=True, fickle_passenger=True)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load_grid()
    episodes = 1000
    df = pd.DataFrame()
    count = 1
    for episode in range(episodes):
        state = env.reset()[0]
        truncated = False
        terminated = False
        steps = 0
        while not truncated and not terminated:
            action = agent.infer(state)
            observation, reward, terminated, truncated, info = env.step(action)
            state = observation
            steps += 1
        if (terminated):
            print(
                f"Episode: {episode + 1}, Steps: {steps}, Count: {count}")
            count += 1
        t = pd.DataFrame(data={'Episode': [episode + 1], 'Steps': [steps]})
        df = pd.concat([df, t])
    df.to_csv('grid_taxi_infer_q.csv')

train()
infer()