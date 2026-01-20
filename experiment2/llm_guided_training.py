import gymnasium as gym
import pandas as pd
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
from problem_solving.dqn_agent import DQNAgent

load_dotenv()
MODEL = 'gpt-4.1-mini'

class RLParams(BaseModel):
    gamma: float = Field('The discount factor')
    epsilon: float = Field('The value that decides the balance between exploration and exploitation')
    epsilon_decay: float = Field('The value that decides the balance decay for exploration phase to allow more exploitation as the training procceds')
    explanation: str = Field('An explanation for proposing the choice of parameters for the RL Agent')
    terminate: bool = Field('A field that determines whether it is time to terminate the training process')



def train():
    env = gym.make('Taxi-v3',max_episode_steps=500, is_rainy=True, fickle_passenger=True)
    agent = DQNAgent(env.observation_space.n, env.action_space.n)
    df = pd.DataFrame()
    agent.gamma = 0.99
    agent.epsilon = 1.0
    agent.epsilon_decay = 0.999

    @function_tool
    def get_training_history():
        return df.to_json(orient='records')

    helper_agent = Agent(
        name="RLDQNAgentHelper",
        model=MODEL,
        instructions="You are reinforcement learning helper agent, that generates hyperparameters for tuning the model."
                     "Your task is to plan and optimize the training process of reinforcement learning agent."
                     "The training history along with previous parameter used and the steps taken to complete the episode "
                     "are available as tools for the model to arrive at optimal hyperparameters."
                     "The hyperparameters must be tuned to reduce the number of steps taken to complete the episode."
                     "You need to optimize parameters as quickly as possible.",
        output_type=RLParams,
        tools=[get_training_history])

    # Train the DQN agent with Experience Replay Buffer
    batch_size = 32
    training_terminate = False
    episode = 1
    while (not training_terminate and episode < 101):
        state,info = env.reset()
        terminated = False
        steps = 0
        agent.exploitation = 0
        agent.exploration = 0
        truncated = False
        while not terminated and not truncated:
            action = agent.act(state)
            observation, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, observation, terminated)
            state = observation
            agent.replay(batch_size)
            steps += 1
        if terminated:
            print(f"Episode: {episode}, Steps : {steps}, 'Exploration': {agent.exploration}, 'Exploitation': {agent.exploitation}")
        t = pd.DataFrame(data={'gamma':[agent.gamma],
                               'epsilon':[agent.epsilon],'epsilon_decay':[agent.epsilon_decay],
                               'Episode': [episode], 'Steps' : [steps], 'Exploration': [agent.exploration], 'Exploitation': [agent.exploitation], 'Terminated': [terminated], 'Truncated': [truncated]})
        df = pd.concat([df,t])

        if (episode % 10 == 0):
            result = Runner.run_sync(starting_agent=helper_agent, input="Please return the next set of hyperparameters in the structured output format.")
            params = result.final_output
            agent.gamma = params.gamma
            agent.epsilon = params.epsilon
            agent.epsilon_decay = params.epsilon_decay
            training_terminate = params.terminate
            print('Explanation : ',params.explanation)

        episode += 1

    df.to_csv('llm_training.csv')
    agent.save_llm()

def infer():
    env = gym.make('Taxi-v3',max_episode_steps=200, is_rainy=True, fickle_passenger=True)
    agent = DQNAgent(env.observation_space.n, env.action_space.n)
    agent.load_llm()
    df = pd.DataFrame()
    num_episodes = 1000
    count = 1
    for episode in range(num_episodes):
        state,info = env.reset()
        terminated = False
        steps = 0
        truncated = False
        while not terminated and not truncated:
            action = agent.infer(state)
            observation, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, observation, terminated)
            state = observation
            steps += 1
        if terminated:
            print(f"Episode: {episode + 1}, Steps: {steps}, Count: {count}")
            count += 1
        t = pd.DataFrame(data={'Episode': [episode + 1], 'Steps': [steps]})
        df = pd.concat([df, t])
    df.to_csv('llm_infer.csv')

train()