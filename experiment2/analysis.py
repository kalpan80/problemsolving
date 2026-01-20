import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['axes.labelsize'] = 14

def exploration():
    df = pd.DataFrame(columns=['Episode', 'Baseline_Exploration', 'Agentic_Exploration'])
    rl = pd.read_csv('taxi_training_q.csv')
    llm = pd.read_csv('llm_training.csv')
    df['Episode'] = llm['Episode']
    df['Baseline_Exploration'] = rl['Exploration']
    df['Agentic_Exploration'] = llm['Exploration']

    df.plot(x='Episode',y=['Baseline_Exploration','Agentic_Exploration'])
    plt.title('Line Plot of Baseline_Exploration and Agentic_Exploration')
    plt.xlabel('Episodes')
    plt.ylabel('Baseline_Exploration and Agentic_Exploration')
    plt.show()

def steps():
    df = pd.DataFrame(columns=['Episode', 'Baseline_Steps', 'Agentic_Steps'])
    rl = pd.read_csv('taxi_infer_q.csv')
    llm = pd.read_csv('llm_infer.csv')
    df['Episode'] = llm['Episode']
    rl.loc[rl['Steps'] == 200, 'Steps'] = 0
    llm.loc[llm['Steps'] == 200, 'Steps'] = 0

    df['Baseline_Steps'] = rl[rl['Steps'] < 200]['Steps']
    df['Agentic_Steps'] = llm[llm['Steps'] < 200]['Steps']

    df.loc[df['Baseline_Steps'] > 0, 'Baseline_Steps'] = 1
    df.loc[df['Agentic_Steps'] > 0, 'Agentic_Steps'] = 1

    df.loc[df['Baseline_Steps'] != 1, 'Baseline_Steps'] = 0
    df.loc[df['Agentic_Steps'] != 1, 'Agentic_Steps'] = 0

    df['Baseline_Steps'] = df['Baseline_Steps'].cumsum()
    df['Agentic_Steps'] = df['Agentic_Steps'].cumsum()

    df.to_csv('steps.csv')

    df.plot(x='Episode',y=['Baseline_Steps','Agentic_Steps'])
    plt.title('Line Plot of Successful Completion of Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Successful Episodes')
    plt.show()


steps()
exploration()