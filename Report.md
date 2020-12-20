# DEEP-RL_Collaboration_and_Competition

In this project, the goal is to control two rackets to pass the ball constantly across the net. If the agent controlling the racket successfully passes the ball across the net it recieves a reward of +0.1. If the ball hits the ground or crosses the boundaries, the agent recieves a reward of -0.01. Rewards will be maximized if the two agents can pass the ball across.


## State and Action Spaces

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement towards (or away from) the net, and jumping. The task is episodic. Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields two different scores. We then take the maximum of these two scores. This yields a single score for each episode. The environment is considered solved if the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents)

## Learning Algorithm

DDPG (Deep Deterministic Policy Gradient) algorithm, which is very suitable for continuous spaces. This algorithm uses two similar neural network, one for the Actor and another for the Critic model.

In the DDPG, the Actor model is trained to represent the polyce itself, in wich is responsable to map states to the best possible actions. It is used to appoximate the optimal policy deterministically. The Critic model, on the other hand, works to learn to evaluate the pair (state, action) predicted by the Actor. That's mean that the both input and output of the actor are used by the critc model, in which the action is the target value.

Since, neural networks as classified as supervised learning, we need the target data in order to train these models. 

he descriptions of the Actor and Critic network are described below:

### Actor:

- Input Layer    : 24 features
- Hidden Layer 1 : 256 neurons
- Hidden Layer 2 : 256 neurons
- Output Layer   : 2 neurons (tanh activation representing two continuous action space - one for movement towards or away from the net and other for jumping)

### Critic:

- Input Layer    : 24 features
- Concat Layer   : (24 from previous layer + 2 from output layer of Actor Network)
- Hidden Layer 1 : 512 neurons
- Output Layer   : 1 (Representing the value of state action pair)


### DDPG Hyper Parameters
- n_episodes (int): maximum number of training episodes
- max_t (int): maximum number of timesteps per episode
- num_agents: number of agents in the environment


Upon running this numerous times it became apparent that the environment would return done at 1000 timesteps. Higher values were irrelevant.

### DDPG Agent Hyper Parameters

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 0.001              # for soft update of target parameters
- LR_ACTOR = 1e-4         # learning rate of the actor
- LR_CRITIC = 1e-4        # learning rate of the critic
- WEIGHT_DECAY = 0   # L2 weight decay
- UPDATE_EVERY = 2        # how often to update the network
- NB_LEARN = 3
- NOISE_DECAY = 0.99
- BEGIN_TRAINING_AT = 500
- NOISE_START = 1.0
- NOISE_END = 0.1


### Neural Networks

The Actor networks utilised two fully connected layers with 256 and 128 units with relu activation and tanh activation for the action space. The network has an initial dimension the same as the state size.

The Critic networks utilised two fully connected layers with 256 and 128 units with leaky_relu activation. The critic network has  an initial dimension the size of the state size plus action size.

## Plot of rewards
![Training](./images/Training.PNG)



## Ideas for Future Work

- Experiment with other algorithms — Tuning the DDPG algorithm required a lot of trial and error. Perhaps another algorithm such as Trust Region Policy Optimization (TRPO), [Proximal Policy Optimization (PPO)](Proximal Policy Optimization Algorithms), or Distributed Distributional Deterministic Policy Gradients (D4PG) would be more robust.
- Add prioritized experience replay — Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare and important experience vectors are sampled.
