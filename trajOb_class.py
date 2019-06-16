"""
Creates a trajectory object from x and y numpy array coordinates
of captured trajectories from videos

@author: Pető Márk
"""

import numpy as np
import itertools
from catmull_rom import catmull_rom
from hparams import spline_resolution
from hparams import checkpoint_division

# TODO create docstrings

class traj_obj():
	def __init__(self,sizeX,sizeY,epsilon,name):

		#trajectory is a numpy array
		#sizeX, sizeY are the initial resolutions, which are further processed
		#epsilon is the width of the confidence interval on the y axis

		# Continuous trajectory interpolated with Centripetal Catmull-Rom spline
		lowerx=np.load('jog_lower_x.npy')
		lowery=np.load('jog_lower_y.npy')
		lowerx=lowerx[1]*sizeX #right hip
		lowery=lowery[1]*sizeY 
		
		#clear x axis offset due to video trimming
		lowerx = lowerx-lowerx[0]

		yratio=sizeY/min(lowery) 
		print("  SizeY//yratio:",sizeY//yratio)

		lowery=lowery-(sizeY//yratio)+2*epsilon
		lowerx=np.rint(lowerx)
		lowery=np.rint(lowery)

		x_intpol,y_intpol = catmull_rom(lowerx, lowery, spline_resolution)
		
		x_intpol = np.rint(x_intpol)
		y_intpol = np.rint(y_intpol)

		#Note that writing coordinates into tuples is a much mroe efficient way handling memory
		coordinates = []
		for i in range(x_intpol.shape[0]):
			coordinates.append((x_intpol[i],y_intpol[i]))

		coordinates = list(coordinates for coordinates,_ in itertools.groupby(coordinates))
		
		coordinates = np.asarray(coordinates)
		print("Interpolated coordinates: ", coordinates.shape)

		self.coordinatesX = np.zeros(coordinates.shape[0])
		self.coordinatesY = np.zeros(coordinates.shape[0])
		for i in range(coordinates.shape[0]):
			self.coordinatesX[i] = coordinates[i][0]
			self.coordinatesY[i] = coordinates[i][1]
		
		self.size=coordinates.shape[0]
		
		self.epsilon=epsilon
		
		self.goal=(lowerx[-1],lowery[-1])
		print("  Goal coordinates:",self.goal)

		self.name=name

		self.checkpoints = []
		for i in range(coordinates.shape[0]):
			if(i%checkpoint_division==0 and i!=0):
				self.checkpoints.append(coordinates[i])
		self.checkpoints = np.array(self.checkpoints)
		
		#print("Checkpoints: ",self.checkpoints)

		print("Trajectory has been created for ",self.name,".")
	
	def get_goal(self):
		return self.goal

	def get_XY(self):
		return self.coordinatesX,self.coordinatesY

	def get_coordinate_by_idx(self,idx):
		return self.coordinatesX[idx],self.coordinatesY[idx]
