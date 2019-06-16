"""
####### EXPERIENCE REPLAY BUFFER ##########
Realizes a dynamically changing buffer memory for temporally experience storage
Experience consists of: current observation, action, reward, target observation (obs_tp1), done

@author: Pető Márk
"""

import numpy as np

class experience_buffer():
	
	def __init__(self,buffer_size):
		"""
		Need to give size explicitly
		:param buffer_size: size of the buffer is non-trivial
		"""
		self.buffer = []
		self.buffer_size = buffer_size
		self.act_idx = 0

	def add(self,experience,total_steps):
		"""
		Add an experience to the buffer, there are 3 semantic cases:
			-  first filling with total_steps and real buffer_size not correlate
			-  continuous rewrite
			-  continue first filling and extending
		:param experience:
		:param total_steps:

		"""
		# if the buffer overflows
		if self.buffer_size <= len(self.buffer) + len(experience):
			if 0 <= self.act_idx < self.buffer_size:
				if total_steps < self.buffer_size:
					self.buffer.extend(experience)
					self.act_idx += 1
				else:
					# loading buffer happens with numpy arrays!
					self.buffer[self.act_idx][0:] = experience
					self.act_idx += 1
			else:
				self.act_idx=0
				self.buffer[self.act_idx][0:] = experience
		# if buffer is filling at the beginning of training
		else:
			self.buffer.extend(experience)
			self.act_idx+=1
			#print(np.array(self.buffer).shape)


	def sample(self,batch_size):
		"""
		Sample a batch of experience for the network

		:param batch_size: size of minibatch
		:return: batch
		"""
		batch = np.zeros((batch_size,5))
		random_idx = np.random.randint(low=0,high=self.buffer_size-batch_size+1)
		batch = np.array(self.buffer[random_idx:random_idx+batch_size][:])
		batch = np.reshape(batch,(32,5))
		#print(batch)
		return batch

	def get_buffer(self):
		return np.array(self.buffer)

	def load_buffer(self,path):
		"""
		Loads buffer into numpy array from .npy saved buffer

		:param path: path for .npy file
		:return: buffer
		"""

		self.buffer = np.load(path,allow_pickle=True)
		self.buffer_size = self.buffer.shape[0]
		print("Buffer has been loaded  with size:",self.buffer_size)

