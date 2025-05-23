
# Reinforcement Learning: Q-Learning and DDQN Agents

This project explores key reinforcement learning (RL) techniques using two well-known environments from the `gymnasium` library: **FrozenLake-v1** and **LunarLander-v2**. The aim is to understand the effects of core hyperparameters on Q-Learning performance and implement a Double Deep Q-Network (DDQN) to solve a continuous control problem.

---

## Overview

This project is divided into two primary parts:

### 1. Q-Learning on Frozen Lake

The FrozenLake-v1 environment provides a discrete and stochastic grid-world challenge ideal for understanding tabular Q-learning. The agent must learn to navigate a slippery frozen lake to reach a goal while avoiding holes.

#### Experiments:
- **Discount Factor (Gamma):** Tested values: `0.01`, `0.2`, `0.5`, `0.99`
- **Learning Rate (Alpha):** Tested values: `0.01`, `0.5`, `0.9`

#### Goals:
- Analyze how gamma affects cumulative reward and value estimation.
- Study how the learning rate impacts convergence and learning stability.
- Provide insights and visualizations comparing different settings.

### 2. Double Deep Q-Network (DDQN) on Lunar Lander

The LunarLander-v2 environment introduces a higher-dimensional state space and continuous dynamics, requiring deep learning-based function approximation.

#### Implementation:
- Designed and implemented a DDQN using **PyTorch**.
- Used experience replay and a target network to stabilize training.
- Tracked episodic rewards to evaluate learning progress.

#### Features:
- Modular DQN code imported from `dqn.py`
- Reward plots for visual performance assessment
- Exploration of the benefits of DDQN over traditional DQN

---

## File Structure

```
.
├── RL_Experiments.ipynb       # Jupyter notebook with code, results, and analysis
├── dqn.py                     # Base DQN code used for Lunar Lander agent
├── ddqn_checkpoint.pth        # Checkpointing the model at the best performance
├── results/
├── ├── Images files           # Results of the models
```

---

## Tools and Libraries

- Python 3.9+
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib / Seaborn
- PyTorch

To install dependencies:

```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

---

## Results and Insights

- Lower gamma values lead to short-sighted policies, while high gamma values prioritize long-term rewards.
- Higher learning rates enable faster updates but may cause instability; moderate rates generally perform best.
- The DDQN agent successfully learns to land the spacecraft by episode ~300–500, demonstrating effective temporal-difference learning and bias mitigation.

---

## Conclusion

This project offers a structured introduction to reinforcement learning, beginning with tabular Q-learning and advancing to neural network-based policies. By comparing algorithm behavior under different settings and solving progressively harder tasks, the project builds a strong foundation in RL concepts and practical implementation.

---
