"""
Creates the environment object

@author: Pető Márk
"""

import numpy as np
from trajOb_class import traj_obj
from agent_class import agent

import hparams

# TODO create docstrings
# TODO stucked agent due to framework error

####### ENVIRONMENT ##########

class traj_env():
	def __init__(self,PMDP,sizeX,sizeY,epsilon):
		self.sizeX=sizeX # final sizeX: trajOb().size + a little
		self.sizeY=sizeY
		self.actions=hparams.n_actions
		self.epsilon=epsilon
		self.trajectory=traj_obj(sizeX,sizeY,epsilon,"Jog pattern,right hip trajectory")

		#Resize environment here:
		coordsX,coordsY=self.trajectory.get_XY()
		self.sizeX=self.sizeX - int((self.sizeX-coordsX[-1])) +10 #to add some extra space to move for the agent, this may cause problems
		self.sizeY=self.sizeY - int((self.sizeY-max(coordsY))) + 2*self.epsilon

		#The agent starts at the beginning of trajectory, (0,0) sebeséggel
		self.agent=agent(posx=self.trajectory.coordinatesX[0],posy=self.trajectory.coordinatesY[0],reward=0.0,velocity=np.zeros(2)) 
		print("Agent's position is: ",self.agent.getPos())

		#self.state_space = [self.agent.X,self.agent.Y,self.agent.velocity[0],self.agent.velocity[1]]
		self.action_space = [0,1,2,3,4,5,6,7]

		#TODO give more info
		self.PMDP=PMDP

		print("Environment has been created for ",self.trajectory.name,".")

	def print_env_properties(self):
		print("Initialized environment has sizeX:",self.sizeX, " and has sizeY:",self.sizeY,"\n","Actions can be taken is:", self.actions,"\n","Environment's confidence interval: ",self.epsilon ,"\n")
	
	#it's easier to check the borders at once in a function
	def checkValidPos(self):
		if( self.agent.X < 0 or self.agent.Y< 0 or self.agent.X>self.sizeX-2 or self.agent.Y>self.sizeY-2):
			return False
		return True

	def reset_env(self):
		#Scenario (epoch) only ends, when the agent reached its goal
		#if(self.check_goal()):
		self.agent.X=self.trajectory.coordinatesX[0] # set position to initial position
		self.agent.Y=self.trajectory.coordinatesY[0]
		self.agent.max_x=self.agent.X
		self.agent.velocity=np.zeros(2)
		#print("Environment has been reset.")
		self.agent.reward = 0.0
		_state = np.zeros(4)
		_state[0] = self.agent.X
		_state[1] = self.agent.Y
		_state[2] = self.agent.velocity[0]
		_state[3] = self.agent.velocity[1]
		return _state

	def update_agent_pos(self):
		#canvas update is not included
		if(self.checkValidPos()):
			nextX,nextY=self.agent.set_pos(self.agent.X+self.agent.velocity[0],self.agent.Y+self.agent.velocity[1])
			return nextX,nextY
		else:
			self.agent.reward=0.0
			self.agent.X=self.trajectory.coordinatesX[0] # set position to initial position
			self.agent.Y=self.trajectory.coordinatesY[0]
			self.agent.max_x=self.trajectory.coordinatesX[0]
			self.agent.velocity=np.zeros(2)
			return self.agent.X,self.agent.Y #If the agent would like to move out of borders, I give back the orginial position instead
		

	def accelerate_agent(self,direction): # agent will learn from 8 differenct accelerating actions
		#directions 0: north 1: north east 2: east 3: south east 4: south 5: south west 6: west 7: north west
		
		#IMPORTANT:can have velocity towards +-inf!

		if direction == 0:
			self.agent.velocity[1]-=1
			#print(" Direction: North")
		if direction == 1:
			self.agent.velocity[0]+=1
			self.agent.velocity[1]-=1
			#print(" Direction: North-East")
		if direction == 2:
			self.agent.velocity[0]+=1
			#print(" Direction: East")
		if direction == 3:
			self.agent.velocity[0]+=1
			self.agent.velocity[1]+=1
			#print(" Direction: South-East")
		if direction == 4:
			self.agent.velocity[1]+=1
			#print(" Direction: South")
		if direction == 5:
			self.agent.velocity[0]-=1
			self.agent.velocity[1]+=1
			#print(" Direction: South-West")
		if direction == 6:
			self.agent.velocity[0]-=1
			#print(" Direction: West")
		if direction == 7:
			self.agent.velocity[0]-=1
			self.agent.velocity[1]-=1
			#print(" Direction: North-West")
		
		#print("  Velocity:",self.agent.velocity)

	def nearest_traj_idx(self):
		agentPos=np.empty(2)
		agentPos[0],agentPos[1]=self.agent.get_pos()
		trajX,trajY=self.trajectory.get_XY()
		ind=0
		for i in range(self.trajectory.coordinatesX.shape[0]):
			#distance calculation-t külön függvénybe tenném!
			if(np.add(np.square(agentPos[0]-trajX[i]),np.square(agentPos[1]-trajY[i])) < np.add(np.square(agentPos[0]-trajX[ind]),np.square(agentPos[1]-trajY[ind]))) :
				ind=i
		
		return ind

	def check_goal(self):
		if(self.agent.get_pos()[0] == self.trajectory.get_goal()[0] and self.agent.get_pos()[1] == self.trajectory.get_goal()[1]):
			self.agent.reward+=1000
			return True
		else:
			return False

	def check_within(self):
		ind=self.nearest_traj_idx()
		trajx,trajy=self.trajectory.get_coordinate_by_idx(ind)
		if((self.agent.Y > trajy-(self.epsilon/2) or self.agent.Y < trajy+(self.epsilon/2)) and self.checkValidPos()):
			return True
		return False

	def check_on_trajectory(self):
		#check whether the agent is on the trajectory points or not, initial point is not included
		agentX=self.agent.get_pos()[0]
		agentY=self.agent.get_pos()[1]

		for i in range(self.trajectory.size):
			if(agentX==int(self.trajectory.get_coordinate_by_idx(i)[0]) and agentY==int(self.trajectory.get_coordinate_by_idx(i)[1])):
				return True
		return False

	def on_checkpoint(self):
		my_bool=False
		for i in range(self.trajectory.checkpoints.shape[0]):
			if(self.agent.max_x < self.trajectory.checkpoints[i][0]):
				if(self.agent.X == self.trajectory.checkpoints[i][0] and 
					(self.agent.Y == self.trajectory.checkpoints[i][1] 
					or (self.agent.Y > self.trajectory.checkpoints[i][1]-(self.epsilon/2) and self.agent.Y < self.trajectory.checkpoints[i][1]+(self.epsilon/2) ))):

					my_bool=True
				else:
					my_bool=False
			else:
				return my_bool

	def step(self,action):
		self.update_agent_pos()
		#print("Position and max_x:",self.agent.X,self.agent.Y,self.agent.max_x)
		_state = np.zeros(4)
		done = self.check_goal()

		if(self.on_checkpoint()):
			self.agent.reward+=10.0

		if(self.check_within()):
			if(self.agent.X > self.agent.max_x):
				self.agent.max_x = self.agent.X
				self.agent.reward += 0.2
		else:
			if(self.checkValidPos()):
				self.agent.reward-=0.8

		#print("Done?: ",done)
		
		if(self.check_on_trajectory()==True and self.agent.X != self.trajectory.coordinatesX[0]):
			self.agent.reward+=1.0

		_state[0] = self.agent.X
		_state[1] = self.agent.Y
		self.accelerate_agent(action)
		_state[2] = self.agent.velocity[0]
		_state[3] = self.agent.velocity[1]
		#print("Reward:",self.agent.reward)
		return _state,self.agent.reward,done

##### EXPERIMENTAL METHODS ######
def episodic_reward(self,num_steps,replay_buffer,episodic_reward_factor): #pretrainReward() alternate
		episodic_reward = int(round(self.agent.X / self.sizeX*episodic_reward_factor))
		#print(self.agent.X / self.sizeX * episodic_reward_factor)
		print("Episodic reward: ",episodic_reward)
		self.agent.reward += episodic_reward

def control_agent(self,direction): 
		#bal felső sarokban van a (0,0)
		#directions 0: north 1: north east 2: east 3: south east 4: south 5: south west 6: west 7: north west
		if direction == 0 and self.agent.Y>=1:
			self.agent.Y-=1
		if direction == 1 and self.agent.Y>=1 and self.agent.X<=self.sizeX-2:
			self.agent.Y-=1
			self.agent.X+=1
		if direction == 2 and self.agent.X<=self.sizeX-2:
			self.agent.X+=1
		if direction == 3 and self.agent.Y<=sizeY-2 and self.agent.X<=self.sizeX-2:
			self.agent.Y+=1
			self.agent.X+=1
		if direction == 4 and self.agent.Y<=self.sizeY-2:
			self.agent.Y+=1
		if direction == 5 and self.agent.Y<=self.sizeY-2 and self.agent.X>=1:
			self.agent.Y+=1
			self.agent.X-=1
		if direction == 6 and self.agent.X>=1:
			self.agent.X-=1
		if direction == 7 and self.agent.X>=1 and self.agent.Y>=1:
			self.agent.Y+=1
			self.agent.X-=1
		print(" New position",self.agent.X,self.agent.Y)

def constrained_accelerate(self,direction): # agent will learn from 8 differenct accelerating actions
		#directions 0: north 1: north east 2: east 3: south east 4: south 5: south west 6: west 7: north west
		#implement actual moving in step() function
		if direction == 0:
			self.agent.velocity[0] = 0.0
			self.agent.velocity[1] = -1.0
			print(" Direction: North")
		if direction == 1:
			self.agent.velocity[0] = 1.0
			self.agent.velocity[1] = -1.0
			print(" Direction: North-East")
		if direction == 2:
			self.agent.velocity[0] = 1.0
			self.agent.velocity[1] = 0.0
			print(" Direction: East")
		if direction == 3:
			self.agent.velocity[0] = 1.0
			self.agent.velocity[1] = 1.0
			print(" Direction: South-East")
		if direction == 4:
			self.agent.velocity[0] = 0.0
			self.agent.velocity[1] = 1.0
			print(" Direction: South")
		if direction == 5:
			self.agent.velocity[0] = -1.0
			self.agent.velocity[1] = 1.0
			print(" Direction: South-West")
		if direction == 6:
			self.agent.velocity[0] = -1.0
			self.agent.velocity[1] = 0.0
			print(" Direction: West")
		if direction == 7:
			self.agent.velocity[0] = -1.0
			self.agent.velocity[1] = -1.0
			print(" Direction: North-West")
		
		print("  Velocity:",self.agent.velocity)

def nearest_traj_idx2(self,posx,posy):
		trajX,trajY=self.trajectory.get_XY()
		ind = 0
		for i in range(trajX.shape[0]):
			if(np.add(np.square(posx-trajX[i]),np.square(posy-trajY[i])) < np.add(np.square(posx-trajX[ind]),np.square(posy-trajY[ind]))) :
				ind = i
		return ind

	
def check_within2(self,posx,posy):
		ind = self.nearest_traj_idx2(posx,posy)
		trajx,trajy=self.trajectory.get_coordinate_by_idx(ind)
		if(posy < trajy-self.epsilon or posy > trajy+self.epsilon):
			return False
		return True
