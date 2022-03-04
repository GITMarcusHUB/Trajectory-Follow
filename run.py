"""
Main code, needs various fixes, TODOs

@author: Pető Márk
"""

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import re
from tkinter import *
import tensorflow.contrib.layers as layers

from catmull_rom import catmull_rom_one_point
from catmull_rom import catmull_rom
#classes
from agent_class import agent
from trajOb_class import traj_obj
from trajEnv_class import traj_env
from Qnetwork_class import Qnetwork
from experience_buffer_class import experience_buffer

import hparams
from update_op import create_op_holder, update_target

is_train = True # Modes: train / evaluation
load_model = False
eval_no = 1

# in test_mode, we replace policy action to a constant [1.0,1.0] acceleration

print("Initializing Environment...")
epsilon_confidence=hparams.epsilon_confidence
env = traj_env(PMDP=True,sizeX=hparams.sizeX,sizeY=hparams.sizeY,epsilon=epsilon_confidence)
env.print_env_properties()
print("Environement has been created.")

print("Creating Tkinter canvas...")
master=Tk()
w = Canvas(master,width=env.sizeX,height=env.sizeY)
w.pack()

img=PhotoImage(width=env.sizeX,height=env.sizeY)

w.create_image((env.sizeX/2,env.sizeY/2),image=img,state="normal") 

for i in range(env.trajectory.coordinatesX.shape[0]):
	img.put("green",(int(env.trajectory.coordinatesX[i]),int(env.trajectory.coordinatesY[i])))

for i in range(env.trajectory.coordinatesX.shape[0]):
	img.put("blue",(int(env.trajectory.coordinatesX[i]),int(env.trajectory.coordinatesY[i])+5))

for i in range(env.trajectory.coordinatesX.shape[0]):
	img.put("blue",(int(env.trajectory.coordinatesX[i]),int(env.trajectory.coordinatesY[i])-5))
print("Canvas is done.")


tf.reset_default_graph()

n_actions = hparams.n_actions
path = hparams.path

q_net = Qnetwork(n_actions=n_actions,learning_rate=hparams.learning_rate)
target_net = Qnetwork(n_actions=n_actions,learning_rate=hparams.learning_rate)

init = tf.global_variables_initializer()

list_of_trainables = tf.trainable_variables()
target_ops = create_op_holder(list_of_trainables,hparams.tau)

replay_buffer = experience_buffer(buffer_size=hparams.buffer_size)

saver = tf.train.Saver()

total_reward_list = []

# Training - Loaing model is missing
if is_train:
	with tf.Session() as sess:
		sess.run(init)
		i = 0

		if load_model:
			print("Loading_model...")
			latest = tf.train.latest_checkpoint(path)
			print("Latest model ckpt:", latest)
			regex = re.compile(r'\d+')
			nums = regex.findall(latest)
			i = int(nums[-1])
			print("Epoch number set to", i)
			i=i+1
			saver.restore(sess, latest)
			replay_buffer.load_buffer(path+"replayBuffer_"+str(i-1)+".npy")
			print(replay_buffer.get_buffer().shape)
			total_steps = hparams.num_steps*(i-1)
			epsilon = hparams.end_epsilon
			total_reward_list = np.load(path+"rewards_2000.npy").tolist()
		else:
			total_steps = 0
			epsilon = hparams.start_epsilon
			lessen_epsilon = (hparams.start_epsilon - hparams.end_epsilon) / hparams.steps_to_lessen_epsilon

		update_target(target_ops,sess)
		while i < hparams.num_episodes+1:
			state = env.reset_env()
			
			if i%10 == 0:
				print("Episode:", i)
			allReward = 0
			done = False
			j=0
			while j <  hparams.num_steps+1:
				j+=1
				# or total_steps < hparams.buffer_size):
				if np.random.randn(1) < epsilon or total_steps < hparams.pre_train_steps:
					action = np.random.randint(0,n_actions)
					if i%10 == 0 and j % hparams.num_steps == 0:
						print("EXPLORATION")
					# print("Action:",action)
					# print("CheckOnTrajectory:",env.checkOnTrajectory())
					# print("CheckWithinConfidence:",env.check_within())
				else:
					action,Q_all = sess.run(fetches=[q_net.predict,q_net.Q_outlayer],
											feed_dict={q_net.inputs : [state],q_net.keep_per : 1.0})
					action=action[0]
					# print("Action:",action)
					# print("CheckOnTrajectory:",env.check_on_trajectory())
					# print("CheckWithinConfidence:",env.check_within())
				state_new,reward,done = env.step(action) 

				# print(np.reshape(np.array([state,action,reward,state_new,done]),[1,5]).shape)
				replay_buffer.add(np.reshape(np.array([state,action,reward,state_new,done]),[1,5]),total_steps=total_steps)
				if epsilon > hparams.end_epsilon and total_steps > hparams.pre_train_steps:
					epsilon -= lessen_epsilon
				if (total_steps > hparams.pre_train_steps and
					total_steps % hparams.train_freq == 0) and \
					total_steps > hparams.buffer_size:
					# prioritization: samoling with probability relative to the last encountered absolute TD-error
					trainBatch = replay_buffer.sample(hparams.batch_size)
					Q1 = sess.run(fetches=q_net.predict,
								  feed_dict={q_net.inputs : np.vstack(trainBatch[:, 3]), q_net.keep_per: 1.0})
					Q2 = sess.run(fetches=target_net.Q_outlayer,
								  feed_dict={target_net.inputs : np.vstack(trainBatch[:,3]),target_net.keep_per : 1.0})
					doubleQ = Q2[range(hparams.batch_size),Q1]
					targetQ = trainBatch[:,2] + (hparams.gamma* doubleQ * -(trainBatch[:, 4] - 1 ))
					_,updateloss,updateerr = sess.run(fetches=[q_net.updateQ, q_net.loss, q_net.err],
													  feed_dict={q_net.inputs : np.vstack(trainBatch[:, 0]),
																 q_net.targetQ : targetQ, q_net.keep_per : 1.0,
																 q_net.actions : trainBatch[:,1]})
					update_target(target_ops,sess)
					if total_steps % 10000 == 0:
						print("updateloss:", updateloss)
						print("updateMaxErr:", updateerr)

				allReward+=reward
				# print("  Reward_gained:",reward)
				state = state_new
				total_steps += 1
				if done == True:
					# print("BREAK")
					break
			# total_reward_list = list(total_reward_list)
			total_reward_list.append(allReward)

			if i % hparams.save_frequeny == 0 and i != 0:
				saver.save(sess,path+str(i)+'.ckpt')
				print("Saved model!")
					
				# save buffer
				if replay_buffer.get_buffer().shape[0] >= hparams.buffer_size:
					np.save(path+"replayBuffer_"+str(i)+".npy",replay_buffer.get_buffer())
				total_reward_list = np.array(total_reward_list)
				np.save(path+"rewards_"+str(i)+".npy",total_reward_list)
				total_reward_list = total_reward_list.tolist()
			# print("Saved rewards!")
			i+=1
# Evaluating:
else:
	with tf.Session() as sess:
		sess.run(init)
		i = 0

		if load_model == True:
			print("Loading_model...")
			latest = tf.train.latest_checkpoint(path)
			print("Latest model ckpt:", latest)
			regex = re.compile(r'\d+')
			nums = regex.findall(latest)
			i = int(nums[-1])
			# LOADS replayBuffer to fully reconstruct learning process:
			# replayBuffer.loadBuffer(path=path+"replayBuffer_"+str(i)+".npy")
			print("Epoch number set to",i)
			i=i+1
			saver.restore(sess,latest)
			replay_buffer.load_buffer(path+"replayBuffer_"+str(i-1)+".npy")
		else:
			print("This is a code for evaluation only...")

		while i < int(nums[-1]) + eval_no + 1:
			state = env.reset_env()
			print("Episode:", i)
			
			allReward = 0
			done = False
			j=0
			while j < hparams.num_steps:
				action, Q_all = sess.run(fetches=[q_net.predict, q_net.Q_outlayer],
										 feed_dict={q_net.inputs: [state], q_net.keep_per: 1.0})
				action = action[0]
				print("action:",action)
				print("check_on_trajectory:",env.check_on_trajectory())
				print("check_within_confidence:",env.check_within())
				state_new, reward, done = env.step(action)
				if env.agent.getPos()[0] > 0 and env.agent.getPos()[1] > 0:
					img.put("red", (int(env.agent.getPos()[0]), int(env.agent.getPos()[1])))

				state = state_new
				print("ACTUAL STEP: ", j)
				j=j+1
			if done:
				print("Done")
				break
			i+=1

w.mainloop()
