"""
Class of the pixel agent

@author: Pető Márk
"""


class agent():
	def __init__(self,posx, posy, reward, velocity):
		"""
		:param posx: x coordinate at given time step
		:param posy: y coordinate at given time step
		:param reward: scalar reward in an episode
		:param velocity: velocity at given time step
		"""
		self.X = posx
		self.Y = posy
		self.reward = reward
		self.velocity = velocity
		# max_x: as far the agent gets during the previous episode
		self.max_x = posx
		print("Agent is created.")
	
	def get_pos(self):
		return self.X, self.Y
	
	def set_pos(self, posx, posy):
		self.X = posx
		self.Y = posy
		return self.X, self.Y

	def get_vel(self):
		return velocity[0], velocity[1]

	def set_vel(self,vel):
		self.velocity = vel
		return self.velocity
