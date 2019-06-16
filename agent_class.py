
class agent():
	def __init__(self,
				 posx,posy,
				 reward,velocity):
		self.X = posx
		self.Y = posy
		self.reward = reward  # reward in a given episode
		self.velocity = velocity
		# max_x: until when it get during the previous episode
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
