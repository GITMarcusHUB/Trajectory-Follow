import numpy as np 
import matplotlib.pyplot as plt

path = "./"

rewards = np.load(path+"rewards_3000.npy")
# rewards2 = np.load(path+"rewards_3000.npy")
# rewards3 = np.load(path+"rewards_4000.npy")
# rewards = np.concatenate((rewards1,rewards2))
# rewards = np.concatenate((rewards,rewards3))

my_loss = [10.792346,0.016147058,159.67413,68.89854,219.4396,
			14.145929,289.49112,358.44022,50.85872,141.59317,
			19.794928,12.455065,0.2093831,1.8738508,7.5043287,
			26.73882,7.6648545,3.0535688,0.02093033,2.051536,
			41.19683,8.277674,0.065252766,1.1669874,0.41933864,
			0.41933864,3.346987e-05,0.28072828,0.0076189004,
			0.00032042526,0.20620531]

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


plt.figure(1)
plt.title("Rewards from 3000 episode")
plt.xlabel("Episodes")
plt.ylabel("Cumulative reward")
plt.plot(rewards,label="maximum reward:"+str(round(max(rewards),2)))
plt.legend()
plt.savefig('./rewards')

plt.show()

