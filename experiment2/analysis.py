import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['axes.labelsize'] = 14

def exploration():
    df = pd.DataFrame(columns=['Episode', 'FixedParams_Exploration', 'GridSearch_Exploration', 'Agentic_Exploration'])
    rl = pd.read_csv('taxi_training_q.csv')
    llm = pd.read_csv('llm_training.csv')
    grid = pd.read_csv('grid_taxi_training_q.csv')

    df['Episode'] = llm['Episode']
    df['FixedParams_Exploration'] = rl['Exploration']
    df['GridSearch_Exploration'] = grid['Exploration']
    df['Agentic_Exploration'] = llm['Exploration']

    df.plot(x='Episode', y=['FixedParams_Exploration', 'GridSearch_Exploration', 'Agentic_Exploration'])
    plt.title('Line Plot for Exploration')
    plt.xlabel('Episodes')
    plt.ylabel('Exploration')
    plt.show()

def steps():
    df = pd.DataFrame(columns=['Episode', 'FixedParams_Steps', 'GridSearch_Steps', 'Agentic_Steps'])
    rl = pd.read_csv('taxi_infer_q.csv')
    llm = pd.read_csv('llm_infer.csv')
    grid = pd.read_csv('grid_taxi_infer_q.csv')

    df['Episode'] = llm['Episode']
    rl.loc[rl['Steps'] == 200, 'Steps'] = 0
    llm.loc[llm['Steps'] == 200, 'Steps'] = 0
    grid.loc[grid['Steps'] == 200, 'Steps'] = 0

    df['FixedParams_Steps'] = rl[rl['Steps'] < 200]['Steps']
    df['GridSearch_Steps'] = grid[grid['Steps'] < 200]['Steps']
    df['Agentic_Steps'] = llm[llm['Steps'] < 200]['Steps']

    df.loc[df['FixedParams_Steps'] > 0, 'FixedParams_Steps'] = 1
    df.loc[df['GridSearch_Steps'] > 0, 'GridSearch_Steps'] = 1
    df.loc[df['Agentic_Steps'] > 0, 'Agentic_Steps'] = 1

    df.loc[df['FixedParams_Steps'] != 1, 'FixedParams_Steps'] = 0
    df.loc[df['GridSearch_Steps'] != 1, 'GridSearch_Steps'] = 0
    df.loc[df['Agentic_Steps'] != 1, 'Agentic_Steps'] = 0

    df['FixedParams_Steps'] = df['FixedParams_Steps'].cumsum()
    df['GridSearch_Steps'] = df['GridSearch_Steps'].cumsum()
    df['Agentic_Steps'] = df['Agentic_Steps'].cumsum()

    df.plot(x='Episode',y=['FixedParams_Steps','GridSearch_Steps','Agentic_Steps'])
    plt.title('Line Plot of Successful Completion of Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Successful Episodes')
    plt.show()


steps()
#exploration()