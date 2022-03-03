# Trajectory-Following
Trajectory following with reinforcement learning, using human keypose as trajectories \
Note: Please note, that this project is no longer actively maintained

## Requirements
`Python 3` \
`Tensorflow 1.15` \
Create a new `virtualenv` \
`pip install requirements/requirements.txt`
## Introduction
The idea behind this project is to try to learn a general trajectory or a set of similar trajectories on a pixel grid with the pixel agent.\
The different strategies, which could be executed by the agent, could tend to learn the different patterns induced by the keyposes.\
The main problem with this approach, that it is not feasible in real-time scenarios.

## Trajectories
Trajectories are created with OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) keypoints. \
The trajectories are independent from each other. The keypoints are placed into a pixel grid, then interpolated using centripetal Catmull-Rom splines.\
The interpolated splines are then used to create a 'race-track' for an agent on the pixel grid, by defining an `epsilon` wide range track from starting point to finish.
The goal is to learn a trajectory path on the racing track, without moving outside from the track.

## Reinforcement Learning with Deep Learning Approach
I have used a pixel-wise agent for this, which made the learning context harder on a large pixel grid.
Maybe a larger agent could have been used to produce an easier task.\

My ambitions for this project was to learn RL algorithms and their connection to a deep learning agent.
The techniques and concepts, that I have used include the following:
- Temporal-difference learning (TD learning)
- One-hot encoding actions
- Discrete state space
- Deep Q network
- Double Q learning
- Huber loss for training
- Experience buffer (agent memory during an epoch/episode)
- Visualization of reward accumulation
- Hyper-parameter fine tuning for a Q-learning agent
- Episodic rewards
- Epsilon-greedy exploration strategy


It is worth to explore more sophisticated methods, which are directly related to the topic of trajectory following. \
A common baseline could be a linear Kalman filter model. Sequential latent variable models could be also investigated for trajectory following, \ 
since the idea to model timedelta as latent variables would be a nice way to create useful model representations for a neural network. 

## Data
Please contact me for the training data, if you are interested in this idea.