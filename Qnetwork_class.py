"""
The Deep Q Network architecture behind the agent

@author: Pető Márk
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Qnetwork():
	
	def __init__(self,n_actions,learning_rate):
		# placeholder shape == (None,4) for the states
		self.inputs = tf.placeholder(shape=[None,4],dtype=tf.float32,name="Input_States") #state: x,y position, x,y velocity
		# placeholder for the changing epsilon-value in e-greedy policy
		self.temp = tf.placeholder(shape=None,dtype=tf.float32,name="temp")
		self.keep_per = tf.placeholder(shape=None,dtype=tf.float32) # for dropout and for q value storage later on

		hidden1 = layers.fully_connected(inputs=self.inputs,num_outputs=64,biases_initializer=None) # default activation_fn is Relu
		hidden2 = layers.fully_connected(inputs=hidden1,num_outputs=128,biases_initializer=None)
		hidden3 = layers.fully_connected(inputs=hidden2,num_outputs=256,biases_initializer=None)
		hidden4 = layers.fully_connected(inputs=hidden3,num_outputs=128,biases_initializer=None)
		hidden5 = layers.fully_connected(inputs=hidden4,num_outputs=64,biases_initializer=None)
		# dropout ???
		self.Q_outlayer = layers.fully_connected(inputs=hidden5,num_outputs=n_actions,activation_fn=None,biases_initializer=None) # 8 action in output, no activation == linear layer, to negative values to be generated
		self.predict = tf.argmax(self.Q_outlayer,1)
		self.Q_dist = tf.nn.softmax(self.Q_outlayer / self.temp) # softmax makes agent select the action
		# creating on_hot vector for the actions
		self.actions = tf.placeholder(shape=[None],dtype=tf.int32,name="Actions_holder") # don't forget: action space is discrete ;)
		self.actions_onehot = tf.one_hot(self.actions,8,dtype=tf.float32)

		#calculating loss by taking the sum of of squares differences between the Q function's TD-target and the actual Q values
		#actual Q values
		self.Q = tf.reduce_sum(tf.multiply(self.Q_outlayer,self.actions_onehot),reduction_indices=1) #get Q values,  reduction_indicies == axis
		#targetQ values:
		self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32,name="targetQ")
		
		self.loss = tf.reduce_sum(self.clipped_error(self.targetQ - self.Q))
		self.err = tf.reduce_max(self.loss)

		trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		#with tf.name_scope("train"):
		self.updateQ = trainer.minimize(self.loss)
	
	#Huber loss
	def clipped_error(self,x):
		try: # delta = 1.0
			return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
		except:
			return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
