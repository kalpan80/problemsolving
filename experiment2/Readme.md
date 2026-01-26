### **Reinforcement Learning Implementation Modules**

* **`dqn_agent.py`**: Implements the Deep Q-Network (DQN) architecture, serving as the core reinforcement learning model for the Taxi-v3 environment.
* **`grid_search_taxi.py`**: A baseline script that performs a traditional grid search to identify optimal hyperparameter configurations through brute-force iteration.
* **`llm_guided_training.py`**: The primary experimental script; it integrates the Agentic AI framework to dynamically tune hyperparameters (e.g., learning rate, epsilon decay) based on real-time training feedback.
* **`taxi_problem.py`**: Establishes a control baseline using fixed, non-optimized parameters to benchmark the performance gains of the Agentic AI approach.
* **`results.txt`**: Comprehensive logs capturing the "Reason-and-Act" (ReAct) chains of the LLM-guided agent during the training phase.
* **`*.csv`**: Structured datasets containing rewards per episode, convergence rates, and final inference metrics for each experimental configuration.