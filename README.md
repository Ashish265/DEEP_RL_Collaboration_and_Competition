# Project DEEP RL Collaboration and Competition

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

### Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started
- Clone the repository using the following command:

- Download python 3.6 and Pytorch 1.0.

- Install the unity environment as described here: [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet/README.md) (The Unity ML-agant environment is already configured by Udacity)

- The current repo has the environment for MAC. If you are using any other operating system download the corresponding build. If so, the code needs to be updated to point to the downloaded environment.

### Training and testing the agent

- The Tennis.ipynb can be executed to train/test the agent. You can skip the training part and directly jump to the last cell to load the saved checkpoints and see the trained agent perform the control task. Load the appropriate environments for the tasks before loading the checkpoints.

