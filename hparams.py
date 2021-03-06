"""
Hyperparameters for the DDQN network

@author: Pető Márk
"""

# TODO variable metrics for some dependent hyperparameters

# Screen resolution parameters & interpolation parameters
sizeX = 640
sizeY = 480
spline_resolution = 13
epsilon_confidence = 10
checkpoint_division = 20

# Training

# appr. 1e4 is good
num_episodes = 2000

# maximum steps for an episode
num_steps = 700
# experience replay buffer size (OpenAI DDQN: 5*10e4,ATARI DDQN: 10e6)
buffer_size = 100000
# 42 000 is 60 episodes
pre_train_steps = 42000
save_frequeny = 500
path = "./"

# parameters
learning_rate = 5e-4
# a.k.a number of experiences in the network
batch_size = 64
# rate of updating target network towards primary Q network, according to OpenAI baselines
tau = 0.0001
# after train_freq steps we update the agent with a TD-update step,
# frequency for updating Q values, the agent can select 8 actions and sense the position and velocity for every state
train_freq = 8

# Learning parameters
n_actions = 8

gamma = .99
start_epsilon = 1
# for epsilon-greedy exploration
end_epsilon = 0.1
# equals to pre_train_steps
steps_to_lessen_epsilon = 42000

# currently not used
episodic_reward_factor = 30
