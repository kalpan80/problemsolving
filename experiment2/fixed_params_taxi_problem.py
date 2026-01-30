import gymnasium as gym

import pandas as pd

from problem_solving.dqn_agent import DQNAgent


def train():
    env = gym.make('Taxi-v3', max_episode_steps=500, is_rainy=True, fickle_passenger=True)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size,action_size)
    episodes = 100
    batch_size = 32
    df = pd.DataFrame()
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
            print(
                f"Episode: {episode + 1}, Steps: {steps}, Exploration: {agent.exploration}, Exploitation: {agent.exploitation}")
        t = pd.DataFrame(data={'gamma':[agent.gamma],
                               'epsilon':[agent.epsilon],'epsilon_decay':[agent.epsilon_decay],
                               'Episode': [episode], 'Steps' : [steps], 'Exploration': [agent.exploration], 'Exploitation': [agent.exploitation], 'Terminated': [terminated], 'Truncated': [truncated]})
        df = pd.concat([df, t])
    df.to_csv('taxi_training_q.csv')
    agent.save()

def infer():
    env = gym.make('Taxi-v3', max_episode_steps=200, is_rainy=True, fickle_passenger=True)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load()
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
#            print(info)
            state = observation
            steps += 1
        if (terminated):
            print(
                f"Episode: {episode + 1}, Steps: {steps}, Count: {count}")
            count += 1
        t = pd.DataFrame(data={'Episode': [episode + 1], 'Steps': [steps]})
        df = pd.concat([df, t])
    df.to_csv('taxi_infer_q.csv')

infer()