# Master Data Science | Institut Polytechnique de Paris

## Capstone Project

<center><img src = "https://www.smeno.com/media/Partenaires/logo-ecoles/logo-institut-polytechnique-paris.png" ></center>

## Topic : Rainbow: Combining Improvements in Deep Reinforcement Learning

### Realised by : 

* Choho Yann Eric CHOHO
* Yedidia AGNIMO

#### Academic year: 2023-2024
February 2024.

## Introduction

The goal of this project is to implement the Rainbow algorithm and compare it to the DQN algorithm. 

The Rainbow algorithm is an improvement of the DQN algorithm. It combines six improvements in deep reinforcement learning. These six improvements are the following:

- Double Q-learning
- Prioritized Experience Replay
- Dueling Network Architecture
- Multi-step Learning
- Distributional RL
- Noisy Nets

## Methodology

We  implement the Rainbow algorithm and compare it to the DQN algorithm and extensions of DQN. We will use the CartPole environment. We will compare the algorithms in terms of a score define in the paper.

The implementation of the Rainbow algorithm is based on the following steps:

1. Importing the required libraries
2. Defining the hyperparameters
3. Defining the agent
4. Defining the replay buffer
5. Defining the network
6. Training the agent
7. Evaluating the agent
8. Visualizing the agent's performance
<!--
9. Saving the agent's model
10. Loading the agent's model
11. Testing the agent's model
-->
[Hessel, Matteo, et al. “Rainbow: Combining Improvements in Deep Reinforcement Learning.” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, Apr. 2018. Crossref](https://arxiv.org/abs/1710.02298)



## 1.Importing Requirements

To run this code, you will need to install the following libraries:

- Open your terminal or command prompt.
- Create a virtual environment (optional but recommended):
  - Run `python -m venv env` to create a virtual environment named "env".
- Activate the virtual environment:
  - On Windows, run `env\Scripts\activate`.
  - On macOS and Linux, run `source env/bin/activate`.
- Install the required libraries by running `pip install -r requirements.txt`.


Les bibliothèques  principale sont  :
```
- torch
- gymnasium (by OpenAI)
```

## 2. Defining the hyperparameters

We define the hyperparameters for the Rainbow algorithm.

```python
# Hyperparameters
BATCH_SIZE = 32
LR = 0.0005
EPSILON = 0.0005
GAMMA = 0.99
TARGET_UPDATE = 1000
REPLAY_MEMORY_SIZE = 15000
LEARNING_STARTS = 1000
N_ATOMS = 51
V_MIN = -10
V_MAX = 10
```

## Structure of the repository
- To launch the code : use the ```notebook.ipynb```

- The `utils` folder contains the `Agent` class used for training each Q-algorithm (including Rainbow). The class name is written in uppercase (e.g., `AGENT`), while the neural network classes have names ending with `Network`. Additionally, there are specific buffer classes with names starting with `Buffer`.

- The `Result` folder contains testing on one episode of each algorithm in .mp4 format
