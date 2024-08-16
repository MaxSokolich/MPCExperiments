import numpy as np
import matplotlib.pyplot as plt
from classes.Learning_module_2d import LearningModule
import math

GP = LearningModule(3)
objective=10
dataset2 = np.load('C:/Users/mahdi/Desktop/MPC/MPCExperiments/datasetGP2.npy')

dataset1 =  np.load('C:/Users/mahdi/Desktop/MPC/MPCExperiments/datasetGP.npy')
GP.read_data_action(dataset1, objective)
GP.read_data_action2(dataset2,objective)


alpha_v_ls =[]
for i in range (len(GP.vy_grid.flatten())): 
    alpha_v = math.atan2(GP.vy_grid.flatten()[i],GP.vx_grid.flatten()[i])
    alpha_v_ls.append(alpha_v)
alpha_v_ls = np.array(alpha_v_ls)
plt.plot(alpha_v_ls)
plt.plot(GP.alpha_grid.flatten())
plt.show()

